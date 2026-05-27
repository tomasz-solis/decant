"""Add Wine tab body — the auth-gated wine-extraction flow.

The biggest tab by surface area. Two input modes (text or photo)
both flow through the same downstream "review and confirm" UX
before writing to Supabase. Auth-gated: anonymous users see a
sign-in nudge instead of the form (closes the OpenAI abuse vector).

`render(history_df, predictor, client, is_authenticated_now)` is
called from inside `with tab1:` in `app.py`. The predictor and the
OpenAI client are passed in rather than re-instantiated here because
both are cached at the app level (`@st.cache_resource`).

The Supabase write client and the cache-clear function are imported
lazily inside the submit handler — same pattern as `tab_palate_maps`.
This keeps the module's import graph minimal and avoids circular
imports back into `app.py`.

This module is large and the original code paths are preserved
verbatim from `app.py`. A follow-up pass could decompose the
`else` branch into smaller functions (input-mode dispatch, review
form, submit handler) — flagged but not done in Chunk 4 to keep
the diff a pure extraction.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from decant.config import OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_SEED
from decant.services.image_storage import save_wine_image
from decant.services.vision_extract import extract_complete_wine_data
from decant.services.wine_match import find_prior_tasting
from decant.supabase_session import get_user_supabase
from decant.ui.helpers import should_display_vintage
from decant.wines_repo import repo_add_wine


def _render_prior_tasting_badge(prior) -> None:
    """Render a one-line badge above the hero score when the wine has
    been tasted before.

    Two visual variants:
    - Exact-vintage match: blue accent ("you've had this exact wine").
    - Different-vintage match: amber accent ("you've had another vintage").

    Kept deliberately small — one line, no expansion. The goal is a
    quick recognition cue, not a full review surface.
    """
    score_part = f"{prior.score:.1f}/10" if prior.score is not None else "no score"
    if prior.liked is True:
        liked_part = "❤️ liked"
    elif prior.liked is False:
        liked_part = "👎 didn't like"
    else:
        liked_part = ""

    pieces = [p for p in [score_part, liked_part] if p]
    rating_text = " · ".join(pieces)

    if prior.match_kind == "exact":
        # Same wine, same vintage — strong recognition.
        bg = "rgba(96, 165, 250, 0.12)"
        border = "rgba(96, 165, 250, 0.35)"
        emoji = "🍷"
        leading = "You've had this"
    else:
        # Same wine, different vintage — softer recognition.
        bg = "rgba(251, 191, 36, 0.12)"
        border = "rgba(251, 191, 36, 0.35)"
        emoji = "🍇"
        vintage_str = str(prior.vintage) if prior.vintage else "another vintage"
        leading = f"You've had the {vintage_str} vintage of this wine"

    st.markdown(
        f"""<div style="
            background: {bg};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 10px 16px;
            margin: 16px 0 0 0;
            font-size: 14px;
            color: #E8E8EB;
        ">
            {emoji} <strong>{leading}</strong> — {rating_text}
        </div>""",
        unsafe_allow_html=True,
    )


def render(
    history_df: pd.DataFrame,
    predictor,
    client,
    is_authenticated_now: bool,
) -> None:
    """Render the Add Wine tab body.

    Args:
        history_df: Already-normalised wine history. Not used when
            unauthenticated.
        predictor: VinoPredictor instance (cached at app level).
        client: OpenAI client (cached at app level). Passed in for
            vision extraction.
        is_authenticated_now: Auth check captured at the top of main().
            Renamed (vs the function `is_authenticated()`) so the
            local usage is clearly a parameter, not a function reference.
    """
    # Local alias preserving the original variable name used inside this
    # function body. Inside the else-branch (the signed-in UI),
    # `is_guest` is logically always False — but the original code
    # threaded it through the save-button disabled state, so the alias
    # keeps that wiring intact rather than removing dead code as part of
    # a pure-extraction pass.
    is_guest = not is_authenticated_now

    # Auth gate: anonymous users see a sign-in nudge instead of the add UI.
    # This closes the OpenAI abuse vector — no Vision API or extraction
    # calls are reachable without a signed-in session.
    if not is_authenticated_now:
        st.markdown("### 🍷 Add Wine to Collection")
        st.info(
            "Sign in to add wines and use the AI extraction feature. "
            "Browsing the gallery and palate maps doesn't require an account."
        )
        st.caption("Use the **Sign in** button at the top right.")
    else:
        st.markdown("### 🍷 Add Wine to Collection")
        st.caption("Enter wine name or upload a photo - AI extracts everything else")

        # `history_df` is the parameter — no reload needed.

        # Input mode selection
        input_mode = st.radio(
            "Input Method",
            ["📝 Enter Wine Name", "📸 Upload Photo"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if input_mode == "📝 Enter Wine Name":
            # Text input mode
            st.markdown("### 🍷 Enter Wine Name")
            st.caption("Type or use voice input (tap microphone on mobile keyboard)")

            wine_name_input = st.text_input(
                "Wine Name",
                placeholder="e.g., Fefiñanes Albariño 2022",
                help="💬 Mobile tip: Use voice input for faster entry!",
                label_visibility="collapsed"
            )

            if wine_name_input and st.button("🔍 CHECK THIS WINE", type="primary", width="stretch"):
                with st.spinner("🧠 AI is extracting wine details from name..."):
                    # predictor was loaded at app boot and passed in.
                    if predictor:
                        extraction = predictor.extract_wine_data(wine_name_input)

                        # Convert to dict
                        wine_data = {
                            'wine_name': extraction.wine_name,
                            'producer': extraction.producer,
                            'vintage': extraction.vintage,
                            'notes': extraction.notes,
                            'score': float(extraction.score),
                            'liked': None,  # User will set
                            'price': 0.0,  # User will set
                            # WINE ORIGIN (AI-extracted)
                            'country': extraction.country,
                            'region': extraction.region,
                            'wine_color': extraction.wine_color,
                            'is_sparkling': extraction.is_sparkling,
                            'is_natural': extraction.is_natural,
                            'sweetness': extraction.sweetness,
                            # Core 5 flavor features
                            'acidity': extraction.acidity,
                            'minerality': extraction.minerality,
                            'fruitiness': extraction.fruitiness,
                            'tannin': extraction.tannin,
                            'body': extraction.body
                        }

                        st.session_state['wine_data'] = wine_data
                        st.success("✅ Wine data extracted!")
                        st.rerun()

        else:
            # Photo upload mode
            st.markdown("### 📸 Snap a Photo")
            st.caption("Point your camera at the wine label - AI does the rest!")

            uploaded_file = st.file_uploader(
                "Tap to open camera or choose photo",
                type=["jpg", "jpeg", "png"],
                help="📱 On mobile: Opens camera automatically | 💻 On desktop: Upload from files",
                label_visibility="visible",
                accept_multiple_files=False
            )

            if uploaded_file:
                # Show image preview
                st.image(uploaded_file, caption="Wine Bottle", width="stretch")

                # Auto-extract ALL data when photo is uploaded
                if 'wine_data' not in st.session_state or st.session_state.get('last_upload') != uploaded_file.name:
                    with st.spinner("🧠 AI is analyzing your wine... extracting all details"):
                        uploaded_file.seek(0)
                        wine_data = extract_complete_wine_data(uploaded_file, history_df, client)

                        if wine_data:
                            st.session_state['wine_data'] = wine_data
                            st.session_state['last_upload'] = uploaded_file.name
                            # Store raw file bytes so we can save the photo later
                            uploaded_file.seek(0)
                            st.session_state['uploaded_photo_bytes'] = uploaded_file.read()
                            st.session_state['uploaded_photo_name'] = uploaded_file.name
                            st.success("✅ Wine analyzed! All fields extracted automatically")
                            st.rerun()

        # Show extracted data if available
        if 'wine_data' in st.session_state:
            wine_data = st.session_state['wine_data']

            # Display wine name prominently with geography
            st.markdown(f"## 🍷 {wine_data['wine_name']}")

            # Location header with NaN-safe fallbacks
            country = wine_data.get('country', None)
            region = wine_data.get('region', None)

            # Convert None, NaN, empty string, or 'nan' string to 'Unknown'
            if country is None or country == '' or str(country).lower() == 'nan' or (isinstance(country, float) and pd.isna(country)):
                country = 'Unknown'
            else:
                country = str(country)

            if region is None or region == '' or str(region).lower() == 'nan' or (isinstance(region, float) and pd.isna(region)):
                region = 'Unknown'
            else:
                region = str(region)

            # Display ONLY if we have real data (no "Unknown" placeholders)
            if country != 'Unknown' and region != 'Unknown':
                st.markdown(f"### 📍 {region}, {country}")
            elif country != 'Unknown':
                st.markdown(f"### 📍 {country}")

            # Style header
            wine_color = wine_data.get('wine_color', 'White')
            region = wine_data.get('region', 'Unknown')
            is_sparkling = wine_data.get('is_sparkling', False)
            sweetness = wine_data.get('sweetness', 'Dry')

            # Build style descriptor
            style_type = "Sparkling" if is_sparkling else "Still"
            style_full = f"{sweetness} {style_type}"

            # Color emojis (used in other sections, not for header)
            color_emoji = {"White": "⚪", "Red": "🔴", "Rosé": "🌸", "Orange": "🟠"}
            color_icon = color_emoji.get(wine_color, '⚪')

            # 🎯 PALATE MATCH VERDICT - Move to TOP (Deep UI Alignment requirement)
            if history_df is not None and len(history_df) > 0:
                # Reuse the predictor passed by the caller (cached at app
                # level) and point it at the latest history.
                predictor.refresh_context(history_df)

                # Calculate likelihood - HARDENED with style-based inference
                wine_features_dict = {
                    'acidity': wine_data.get('acidity', 0),
                    'minerality': wine_data.get('minerality', 0),
                    'fruitiness': wine_data.get('fruitiness', 0),
                    'tannin': wine_data.get('tannin', 0),
                    'body': wine_data.get('body', 0)
                }

                # 🚨 If features not extracted from image, use OpenAI to infer with explanation
                feature_descriptions = {}
                if all(v == 0 for v in wine_features_dict.values()):
                    wine_name = wine_data.get('wine_name', '')
                    region = wine_data.get('region', 'Unknown')

                    # Ask OpenAI to rate AND explain each characteristic
                    st.info("ℹ️ Wine characteristics inferred from wine name and region (not extracted from label)")

                    # Cache key for consistent results
                    cache_key = f"{wine_name}_{region}".lower().replace(" ", "_")

                    # Check if we've already rated this wine
                    if 'wine_ratings_cache' not in st.session_state:
                        st.session_state['wine_ratings_cache'] = {}

                    if cache_key in st.session_state['wine_ratings_cache']:
                        # Use cached ratings for consistency
                        cached = st.session_state['wine_ratings_cache'][cache_key]
                        wine_features_dict = cached['features']
                        feature_descriptions = cached['descriptions']
                        wine_data.update({
                            'acidity': wine_features_dict['acidity'],
                            'fruitiness': wine_features_dict['fruitiness'],
                            'body': wine_features_dict['body'],
                            'minerality': wine_features_dict['minerality'],
                            'tannin': wine_features_dict['tannin']
                        })
                        st.caption("✓ Using cached ratings for consistency")
                    else:
                        # First time - get ratings from LLM
                        try:
                            # Nuclear-Grade Feature Extraction Prompt for Decision Science
                            inference_prompt = f"""Role: You are a Master Sommelier and Data Engineer specializing in quantitative viticulture.

Task: Provide a precise, technical flavor profile for the wine: {wine_name} from {region}.

Objective: Your output will be used to calculate a vector-space similarity model. Consistency in your scoring logic is mandatory.

Scoring Guidelines (Scale 1.0 - 10.0):
• Acidity: 1.0 (Flat/Flabby) to 10.0 (High Tartaric/Piercing)
• Fruitiness: 1.0 (Earth-driven/Savory) to 10.0 (Primary Fruit Bomb/Jammy)
• Body: 1.0 (Light/Watery) to 10.0 (Full/Viscous/Heavy)
• Tannin: 1.0 (No structure/Silk) to 10.0 (Aggressive/Gripping/Astringent)
• Minerality: 1.0 (Clean/Fruit-only) to 10.0 (Stony/Saline/Chalky)

Requirements:
1. Use your internal knowledge of this specific producer, vintage, and regional style.
2. Avoid "safe" middle-ground scores (like 5.0) unless truly warranted.
3. Provide the output ONLY as a JSON object for programmatic parsing.

Desired JSON Structure:
{{
  "wine_metadata": {{
    "name": "{wine_name}",
    "region": "{region}",
    "style": "Regional style description"
  }},
  "technical_profile": {{
    "acidity": float,
    "fruitiness": float,
    "body": float,
    "tannin": float,
    "minerality": float
  }},
  "sommelier_verdict": "One sentence technical summary of the structure."
}}"""

                            response = client.chat.completions.create(
                                model=OPENAI_MODEL,
                                messages=[
                                    {"role": "user", "content": inference_prompt}
                                ],
                                response_format={"type": "json_object"},
                                temperature=OPENAI_TEMPERATURE,
                                seed=OPENAI_SEED
                            )

                            import json
                            from pydantic import ValidationError
                            from decant.constants import LLMWineAnalysis

                            # Parse JSON response
                            result = json.loads(response.choices[0].message.content)

                            # SECURITY FIX: Validate LLM response with Pydantic
                            try:
                                validated_response = LLMWineAnalysis.model_validate(result)

                                # Extract technical profile scores from validated response
                                profile = validated_response.technical_profile
                                wine_features_dict = {
                                    'acidity': float(profile.acidity),
                                    'fruitiness': float(profile.fruitiness),
                                    'body': float(profile.body),
                                    'minerality': float(profile.minerality),
                                    'tannin': float(profile.tannin)
                                }

                                # Use sommelier verdict as explanation for all features
                                sommelier_verdict = validated_response.sommelier_verdict
                                feature_descriptions = {
                                    'acidity': f"{profile.acidity}/10 - {sommelier_verdict}",
                                    'fruitiness': f"{profile.fruitiness}/10 - {sommelier_verdict}",
                                    'body': f"{profile.body}/10 - {sommelier_verdict}",
                                    'minerality': f"{profile.minerality}/10 - {sommelier_verdict}",
                                    'tannin': f"{profile.tannin}/10 - {sommelier_verdict}"
                                }

                                # Update wine_data with inferred values so they display correctly
                                wine_data['acidity'] = wine_features_dict['acidity']
                                wine_data['fruitiness'] = wine_features_dict['fruitiness']
                                wine_data['body'] = wine_features_dict['body']
                                wine_data['minerality'] = wine_features_dict['minerality']
                                wine_data['tannin'] = wine_features_dict['tannin']

                                # Cache the results for future consistency
                                st.session_state['wine_ratings_cache'][cache_key] = {
                                    'features': wine_features_dict,
                                    'descriptions': feature_descriptions
                                }

                            except ValidationError as ve:
                                # Validation failed - LLM returned invalid data
                                st.error(f"🚨 LLM returned invalid response structure: {ve}")
                                st.info("💡 Please enter features manually below.")
                                # Don't cache invalid results

                        except json.JSONDecodeError as je:
                            st.error(f"🚨 LLM returned invalid JSON: {je}")
                            st.info("💡 Please enter features manually below.")
                        except KeyError as ke:
                            st.error(f"🚨 LLM response missing required field: {ke}")
                            st.info("💡 Please enter features manually below.")
                        except Exception as e:
                            st.warning(f"⚠️ Could not infer wine characteristics: {str(e)}")
                            st.info("💡 Please enter features manually below.")
                            wine_features_dict = None

                # 🎯 PALATE ENGINE - SINGLE SOURCE OF TRUTH
                # display_match_score is what shows in the hero card. As of
                # the 2026-05 display fix, this is the *flavor alignment*
                # number (palate_match), not the multiplied likelihood.
                # Confidence is shown separately in the breakdown panel so
                # users can read both facts independently.
                palate_score = None
                display_match_score = None

                if wine_features_dict is not None:
                    palate_score = predictor.calculate_palate_score(
                        wine_features_dict,
                        wine_color
                    )
                    # Headline = flavor alignment. The old likelihood_score
                    # (palate_match * confidence) is still on PalateScore
                    # for anyone who wants it, but the user-facing number
                    # is the alignment.
                    display_match_score = palate_score.palate_match

                # 🍷 PRIOR TASTING BADGE — appears above the hero card
                # if the user has had this wine (or a different vintage
                # of it) before. Token-based matching with producer as
                # the gate; see decant.services.wine_match.
                prior = find_prior_tasting(
                    candidate_name=wine_data.get('wine_name', ''),
                    candidate_producer=wine_data.get('producer'),
                    candidate_vintage=wine_data.get('vintage'),
                    history_df=history_df,
                )
                if prior is not None:
                    _render_prior_tasting_badge(prior)

                # 🎯 HERO CARD: Palate Recommendation Score (SOLE AUTHORITATIVE DISPLAY)
                # CHECK: Display score only if it exists AND is calculated (not None, not just initialized)
                if display_match_score is not None and palate_score is not None:
                    # DISPLAY: Show the actual calculated score (even if 0, it's a real calculation)
                    # MOBILE-OPTIMIZED: Larger text, clearer verdict for in-shop quick glance
                    st.markdown(f"""
<div class="glass-card glow" style="text-align: center; padding: 32px 24px; margin: 20px 0; position: relative;">
    <p style="color: #A0A0A8; margin: 0 0 12px 0; font-size: clamp(10px, 2.5vw, 12px); text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;">
        Palate Recommendation Score
    </p>
    <div class="match-score-gradient" style="font-size: clamp(60px, 15vw, 80px); margin: 0; font-family: 'Geist', 'Inter', sans-serif; line-height: 1;">
        {display_match_score:.1f}%
    </div>
    <p style="color: #E8E8EB; margin: 12px 0 0 0; font-size: clamp(14px, 4vw, 18px); font-weight: 600;">{palate_score.verdict}</p>
</div>
""", unsafe_allow_html=True)

                    # Breakdown panel: alignment + confidence as two
                    # separate facts. No "x = y" formula — that's the
                    # framing that made users read low-confidence scores
                    # as a regression. Confidence is shown so the user
                    # can read both numbers but isn't asked to multiply
                    # them mentally.
                    n_samples = palate_score.n_samples
                    confidence_pct = palate_score.confidence_factor * 100
                    if confidence_pct >= 90:
                        confidence_label = "high"
                    elif confidence_pct >= 60:
                        confidence_label = "moderate"
                    else:
                        confidence_label = "low — based on a small sample"

                    wines_to_95 = max(0, 10 - n_samples)
                    tip_html = (
                        f"💡 Add <strong style=\"color: #E8E8EB;\">{wines_to_95} more wine(s)</strong> "
                        f"to reach 95%+ confidence."
                        if wines_to_95 > 0
                        else "💡 Your collection is large enough for high-confidence recommendations."
                    )

                    st.markdown(f"""<div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0;">
<p style="color: #A0A0A8; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin: 0 0 1rem 0;">🔍 What This Score Means</p>
<div style="margin-bottom: 1rem;">
<p style="color: #E8E8EB; font-weight: 700; font-size: 14px; margin: 0 0 4px 0;">Flavor Alignment: <span style="color: #800020;">{palate_score.palate_match:.1f}%</span></p>
<p style="color: #A0A0A8; font-size: 12px; margin: 0; line-height: 1.5;">How similar this wine's flavor profile is to wines you've enjoyed. This is the headline number above.</p>
</div>
<div style="margin-bottom: 1rem;">
<p style="color: #E8E8EB; font-weight: 700; font-size: 14px; margin: 0 0 4px 0;">Confidence: <span style="color: #800020;">{confidence_label}</span></p>
<p style="color: #A0A0A8; font-size: 12px; margin: 0; line-height: 1.5;">Based on {n_samples} wine(s) you've rated as liked. More ratings = more confident recommendations.</p>
</div>
<p style="color: #A0A0A8; font-size: 11px; margin: 12px 0 0 0; line-height: 1.6;">{tip_html}</p>
</div>""", unsafe_allow_html=True)
                else:
                    # LOADING STATE: Show "Calculating..." text instead of 0%
                    st.markdown("""
<div class="glass-card glow" style="text-align: center; padding: 40px 30px; margin: 24px 0;">
    <p style="color: #A0A0A8; margin: 0 0 16px 0; font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;">Palate Recommendation Score</p>
    <div class="match-score-gradient" style="font-size: 48px; margin: 16px 0; font-family: 'Geist', 'Inter', sans-serif;">
        Calculating...
    </div>
    <p style="color: #A0A0A8; margin: 16px 0 0 0; font-size: 14px;">Analysing your palate profile</p>
</div>
""", unsafe_allow_html=True)

                # Add visual separator
                st.markdown("---")

                # 📋 CLEAN PROFESSIONAL PRESENTATION - 2-Column Layout
                st.markdown("### 📋 Wine Profile")

                eval_col1, eval_col2 = st.columns(2)

                # LEFT COLUMN: Style, Origin, Vintage
                with eval_col1:
                    st.markdown("**🍷 Style & Origin**")
                    # Vertical bulleted list format - clean hierarchy
                    st.markdown(f"- **Type:** {wine_color}")
                    st.markdown(f"- **Style:** {style_full}")
                    # Show Appellation with region hierarchy
                    if region != 'Unknown' and country != 'Unknown':
                        st.markdown(f"- **Appellation:** {region} ({country})")
                    elif region != 'Unknown':
                        st.markdown(f"- **Appellation:** {region}")
                    elif country != 'Unknown':
                        st.markdown(f"- **Origin:** {country}")
                    if should_display_vintage(wine_data.get('vintage')):
                        st.markdown(f"- **Vintage:** {int(wine_data.get('vintage'))}")
                    if wine_data.get('producer'):
                        st.markdown(f"- **Producer:** {wine_data.get('producer')}")

                # RIGHT COLUMN: Tasting Notes & Verdict
                with eval_col2:
                    st.markdown("**📝 Tasting Notes & Verdict**")
                    notes = wine_data.get('notes', 'No tasting notes available')

                    # Display full notes with natural wrapping (no truncation)
                    st.markdown(f"_{notes}_")

                    # Why you'll like it — copy adapts to the alignment
                    # score (the new headline). Thresholds mirror the
                    # engine's STRONG_ALIGNMENT (70) and EXPLORE_ALIGNMENT (55).
                    st.markdown("")  # spacing
                    if display_match_score is not None:
                        if display_match_score >= 70:
                            why_like = f"**💙 Why you'll like it:** This matches your preferred {wine_color.lower()} style closely."
                        elif display_match_score >= 55:
                            why_like = f"**🧡 Why try it:** Reasonable compatibility with your palate, worth exploring."
                        else:
                            why_like = f"**🟡 Different:** This is a departure from your usual {wine_color.lower()} wines."
                        st.markdown(why_like)

                st.markdown("---")
            else:
                st.info("🔍 Add wines to your collection to see palate match predictions")
                st.markdown("---")

            # 95% PRE-POPULATED "STORE MODE" UI
            st.markdown("### 💾 Store Mode - Quick Log")
            st.caption("AI extracted everything - only 3 inputs needed from you!")

            # OPTIMIZED FORM: 3 inputs in one clean row [Score, Price, Like-Toggle]
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                # Score (slider for quick input)
                score_input = st.slider(
                    "⭐ Your Score",
                    min_value=1.0,
                    max_value=10.0,
                    value=float(wine_data.get('score', 7.5)),
                    step=0.5,
                    help="How would you rate this wine?"
                )

            with col2:
                # Price - moved from Technical Details for better UX
                price_input = st.number_input(
                    "💶 Price (€)",
                    min_value=0.0,
                    value=float(wine_data.get('price', 0.0)),
                    step=0.50,
                    help="Retail price in EUR"
                )

            with col3:
                # Liked toggle: smart default based on alignment.
                # 55 mirrors the engine's EXPLORE_ALIGNMENT — solidly
                # positive alignment territory. Conservative on
                # purpose: better to default to "not yet liked" and
                # have the user opt in than oversell.
                if display_match_score is not None:
                    liked_default = display_match_score >= 55
                else:
                    # Fallback for truly empty history: neutral default
                    liked_default = (score_input >= 7.0)

                liked_input = st.toggle(
                    "❤️ Did You Like It?",
                    value=liked_default,
                    help="Would you buy this again?"
                )

            # Advanced details in expander (AI-extracted technical data)
            with st.expander("⚙️ Technical Details & Edit Data (Optional)"):
                st.markdown("#### 🎯 Flavor Profile (0-10 Scale)")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("⚡ Acidity", f"{wine_data['acidity']}/10")
                with col2:
                    st.metric("💎 Minerality", f"{wine_data['minerality']}/10")
                with col3:
                    st.metric("🍇 Fruitiness", f"{wine_data['fruitiness']}/10")
                with col4:
                    st.metric("🌰 Tannin", f"{wine_data['tannin']}/10")
                with col5:
                    st.metric("💪 Body", f"{wine_data['body']}/10")

                # Show explanations if features were inferred (not extracted from image)
                if feature_descriptions:
                    st.markdown("")
                    st.markdown("**📝 Characteristic Explanations:**")
                    st.markdown(f"• **Acidity ({wine_data['acidity']}/10)**: {feature_descriptions.get('acidity', 'N/A')}")
                    st.markdown(f"• **Fruitiness ({wine_data['fruitiness']}/10)**: {feature_descriptions.get('fruitiness', 'N/A')}")
                    st.markdown(f"• **Body ({wine_data['body']}/10)**: {feature_descriptions.get('body', 'N/A')}")
                    st.markdown(f"• **Minerality ({wine_data['minerality']}/10)**: {feature_descriptions.get('minerality', 'N/A')}")
                    st.markdown(f"• **Tannin ({wine_data['tannin']}/10)**: {feature_descriptions.get('tannin', 'N/A')}")

                st.markdown("---")

                st.markdown("#### 📊 Full Technical Specifications")
                tech_col1, tech_col2 = st.columns(2)
                with tech_col1:
                    st.markdown(f"**Wine Color:** {wine_data.get('wine_color', 'White')}")
                    st.markdown(f"**Sparkling:** {'Yes' if wine_data.get('is_sparkling', False) else 'No'}")
                    st.markdown(f"**Natural:** {'Yes' if wine_data.get('is_natural', False) else 'No'}")
                with tech_col2:
                    st.markdown(f"**Sweetness:** {wine_data.get('sweetness', 'Dry')}")
                    st.markdown(f"**Producer:** {wine_data.get('producer', 'Unknown')}")
                    if should_display_vintage(wine_data.get('vintage')):
                        st.markdown(f"**Vintage:** {int(wine_data.get('vintage'))}")
                    else:
                        st.markdown(f"**Vintage:** NV")

            # Large, prominent Save button (login required)
            if is_guest:
                st.warning("🔒 Log in to save wines to your collection")

            if st.button("💾 SAVE TO MY COLLECTION", type="primary", width="stretch", disabled=is_guest):
                # Validate and update user inputs
                try:
                    # Type validation with high-dimensional attributes
                    wine_data['score'] = float(score_input)
                    wine_data['liked'] = bool(liked_input)  # Ensure boolean
                    wine_data['price'] = float(price_input)  # Price is now always in Quick Log

                    # Input validation - catch invalid data early
                    validation_errors = []

                    if not wine_data.get('wine_name') or wine_data['wine_name'].strip() == '':
                        validation_errors.append("Wine name is required")

                    if wine_data['score'] < 1 or wine_data['score'] > 10:
                        validation_errors.append(f"Score must be 1-10 (got {wine_data['score']})")

                    if wine_data['price'] < 0:
                        validation_errors.append(f"Price cannot be negative (got {wine_data['price']})")

                    # Validate flavor features (must be 1-10)
                    for feature in ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']:
                        value = wine_data.get(feature, 0)
                        if value < 1 or value > 10:
                            validation_errors.append(f"{feature.capitalize()} must be 1-10 (got {value})")

                    if validation_errors:
                        st.error(f"🚫 Cannot save wine - please fix these issues:\n" + "\n".join(f"• {err}" for err in validation_errors))
                        st.stop()

                    # Validate high-dimensional fields
                    wine_data['is_sparkling'] = bool(wine_data.get('is_sparkling', False))
                    wine_data['is_natural'] = bool(wine_data.get('is_natural', False))

                    # Save to Supabase wines table (RLS-authenticated session)
                    row_data = {
                        'wine_name': wine_data['wine_name'],
                        'producer': wine_data['producer'],
                        'vintage': wine_data['vintage'],
                        'notes': wine_data['notes'],
                        'score': wine_data['score'],
                        'liked': wine_data['liked'],
                        'price': wine_data['price'],
                        # WINE ORIGIN
                        'country': wine_data.get('country', 'Unknown'),
                        'region': wine_data.get('region', 'Unknown'),
                        # HIGH-DIMENSIONAL ATTRIBUTES
                        'wine_color': wine_data.get('wine_color', 'White'),
                        'is_sparkling': wine_data['is_sparkling'],
                        'is_natural': wine_data['is_natural'],
                        'sweetness': wine_data.get('sweetness', 'Dry'),
                        # Core 5 flavor features
                        'acidity': wine_data['acidity'],
                        'minerality': wine_data['minerality'],
                        'fruitiness': wine_data['fruitiness'],
                        'tannin': wine_data['tannin'],
                        'body': wine_data['body']
                    }

                    try:
                        with st.spinner("💾 Saving wine to Supabase..."):
                            repo_add_wine(get_user_supabase(), row_data)
                        st.success("✅ Wine saved to Supabase!")
                    except Exception as supabase_error:
                        st.error(f"❌ Supabase error while saving wine: {supabase_error}")
                        st.stop()

                    # Save uploaded photo if available
                    photo_bytes = st.session_state.get('uploaded_photo_bytes')
                    photo_name = st.session_state.get('uploaded_photo_name')
                    if photo_bytes and wine_data.get('wine_name'):
                        photo_file = io.BytesIO(photo_bytes)
                        photo_file.name = photo_name or "photo.jpg"
                        saved_path = save_wine_image(photo_file, wine_data['wine_name'])
                        if saved_path:
                            st.info("📸 Photo saved")

                    # Invalidate the load_wine_data cache so the next
                    # read sees the write. Imported lazily to avoid a
                    # circular import on app.py.
                    from app import clear_wine_data_cache
                    clear_wine_data_cache()

                    st.success(f"✅ Saved {wine_data['wine_name']} to your collection!")
                    st.balloons()

                    # Clear session state to start fresh
                    for key in ['wine_data', 'last_upload', 'uploaded_photo_bytes', 'uploaded_photo_name']:
                        st.session_state.pop(key, None)

                    st.info("🍷 Ready for next wine! Add another above.")

                except ValueError as e:
                    st.error(f"Validation error: {str(e)}")
                    st.info("Please check that price is a valid number and liked is true/false")
                except Exception as e:
                    st.error(f"Error saving: {str(e)}")
                    st.info("Check Supabase configuration and RLS permissions")

            else:
                # No data extracted yet
                st.info("👆 Enter a wine name or upload a photo to get started")

