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

from decant.services.image_storage import save_wine_image
from decant.services.text_infer import infer_features_from_text
from decant.services.vision_extract import extract_complete_wine_data
from decant.services.wine_match import find_prior_tasting
from decant.supabase_session import get_user_supabase
from decant.ui.helpers import should_display_vintage
from decant.wines_repo import repo_add_wine


def _render_prior_tasting_badge(prior) -> None:
    """Render a one-line badge above the hero score when the wine has
    been tasted before.

    Two visual variants:
    - Exact-vintage match: olive accent ("you've had this exact wine").
    - Different-vintage match: terracotta accent ("you've had another vintage").

    Kept deliberately small — one line, no expansion. The goal is a
    quick recognition cue, not a full review surface.
    """
    score_part = f"{prior.score:.1f}/10" if prior.score is not None else "no score"
    if prior.liked is True:
        liked_part = "liked"
    elif prior.liked is False:
        liked_part = "didn't like"
    else:
        liked_part = ""

    pieces = [p for p in [score_part, liked_part] if p]
    rating_text = " · ".join(pieces)

    if prior.match_kind == "exact":
        # Same wine, same vintage: strong recognition.
        bg = "var(--olive-soft)"
        border = "var(--olive)"
        leading = "You've had this"
    else:
        # Same wine, different vintage: softer recognition.
        bg = "var(--terracotta-soft)"
        border = "var(--terracotta)"
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
            color: var(--text-primary);
        ">
            <strong>{leading}</strong> — {rating_text}
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
        st.markdown("## Add Wine to Collection")
        st.info(
            "Sign in to add wines and use the AI extraction feature. "
            "Browsing the gallery and palate maps doesn't require an account."
        )
        st.caption("Use the **Sign in** button at the top right.")
    else:
        st.markdown("## Add Wine to Collection")
        st.caption("Enter wine name or upload a photo - AI extracts everything else")

        # `history_df` is the parameter — no reload needed.

        # Input mode selection
        input_mode = st.radio(
            "Input Method",
            ["Enter Wine Name", "Upload Photo"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if input_mode == "Enter Wine Name":
            # Text input mode
            st.markdown("### Enter Wine Name")
            st.caption("Type or use voice input (tap microphone on mobile keyboard)")

            wine_name_input = st.text_input(
                "Wine Name",
                placeholder="e.g., Fefiñanes Albariño 2022",
                help="Mobile tip: Use voice input for faster entry!",
                label_visibility="collapsed"
            )

            if wine_name_input and st.button("CHECK THIS WINE", type="primary", width="stretch"):
                with st.spinner("AI is extracting wine details from name..."):
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

                        # If the structured extraction didn't produce a
                        # flavour profile (common for text entry — the
                        # name-based extraction focuses on metadata), infer
                        # it ONCE here, at extraction time. Doing it now
                        # rather than at display time is the fix for the
                        # unstable-score bug: the display path reran the
                        # inference on every Streamlit rerun, and the LLM
                        # isn't truly deterministic, so the score drifted
                        # between renders. Inferred once and frozen, the
                        # score is stable.
                        flavor_features = [
                            wine_data['acidity'], wine_data['minerality'],
                            wine_data['fruitiness'], wine_data['tannin'],
                            wine_data['body'],
                        ]
                        if all((v or 0) == 0 for v in flavor_features):
                            inferred = infer_features_from_text(
                                wine_name=wine_data.get('wine_name', ''),
                                region=wine_data.get('region', 'Unknown') or 'Unknown',
                                client=client,
                            )
                            if inferred is not None:
                                wine_data.update(inferred)

                        st.session_state['wine_data'] = wine_data
                        st.success("Wine data extracted.")
                        st.rerun()

        else:
            # Photo upload mode
            st.markdown("### Snap a Photo")
            st.caption("Point your camera at the wine label - AI does the rest!")

            uploaded_file = st.file_uploader(
                "Tap to open camera or choose photo",
                type=["jpg", "jpeg", "png"],
                help="On mobile: opens camera automatically. On desktop: upload from files.",
                label_visibility="visible",
                accept_multiple_files=False
            )

            if uploaded_file:
                # Show image preview
                st.image(uploaded_file, caption="Wine Bottle", width="stretch")

                # Auto-extract ALL data when photo is uploaded
                if 'wine_data' not in st.session_state or st.session_state.get('last_upload') != uploaded_file.name:
                    with st.spinner("AI is analyzing your wine and extracting details..."):
                        uploaded_file.seek(0)
                        wine_data = extract_complete_wine_data(uploaded_file, history_df, client)

                        if wine_data:
                            st.session_state['wine_data'] = wine_data
                            st.session_state['last_upload'] = uploaded_file.name
                            # Store raw file bytes so we can save the photo later
                            uploaded_file.seek(0)
                            st.session_state['uploaded_photo_bytes'] = uploaded_file.read()
                            st.session_state['uploaded_photo_name'] = uploaded_file.name
                            st.success("Wine analyzed. All fields extracted automatically.")
                            st.rerun()

        # Show extracted data if available
        if 'wine_data' in st.session_state:
            wine_data = st.session_state['wine_data']

            # Display wine name prominently with geography
            st.markdown(f"## {wine_data['wine_name']}")

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

            # Location line: NOT a section heading. Semantically it is
            # metadata under the wine title, so render it as a styled
            # div with body font.
            if country != 'Unknown' and region != 'Unknown':
                location_text = f"{region}, {country}"
            elif country != 'Unknown':
                location_text = country
            else:
                location_text = None

            if location_text:
                st.markdown(
                    f"<div style='font-family: var(--font-body); "
                    f"font-size: 1.05rem; font-weight: 600; "
                    f"color: var(--text-secondary); margin: 4px 0 16px 0;'>"
                    f"{location_text}</div>",
                    unsafe_allow_html=True,
                )

            # Style header
            wine_color = wine_data.get('wine_color', 'White')
            region = wine_data.get('region', 'Unknown')
            is_sparkling = wine_data.get('is_sparkling', False)
            sweetness = wine_data.get('sweetness', 'Dry')

            # Build style descriptor
            style_type = "Sparkling" if is_sparkling else "Still"
            style_full = f"{sweetness} {style_type}"

            # Palate match verdict: keep it above the details.
            if history_df is not None and len(history_df) > 0:
                # Reuse the predictor passed by the caller (cached at app
                # level) and point it at the latest history.
                predictor.refresh_context(history_df)

                # Read the (already-frozen) flavour features from the
                # stored wine record. Inference happened ONCE at
                # extraction time (see the text-entry path and
                # services/text_infer). The display path must never call
                # the LLM — that was the source of the unstable score,
                # because reruns re-inferred and the LLM isn't truly
                # deterministic.
                wine_features_dict = {
                    'acidity': wine_data.get('acidity', 0),
                    'minerality': wine_data.get('minerality', 0),
                    'fruitiness': wine_data.get('fruitiness', 0),
                    'tannin': wine_data.get('tannin', 0),
                    'body': wine_data.get('body', 0)
                }

                # If features are still all zero, inference failed at
                # extraction time (LLM unavailable, etc). Don't retry
                # here — show a manual-entry prompt and skip scoring.
                if all((v or 0) == 0 for v in wine_features_dict.values()):
                    st.info(
                        "No flavour profile available for this wine. Enter "
                        "the characteristics manually below to see a palate match."
                    )
                    wine_features_dict = None

                # Palate engine: single source of truth.
                # display_match_score is what shows in the hero card. As of
                # the 2026-05 display fix, this is the *flavor alignment*
                # number (palate_match), not the multiplied likelihood.
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

                # Prior tasting badge appears above the hero card
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

                # Hero card: palate recommendation score.
                # CHECK: Display score only if it exists AND is calculated (not None, not just initialized)
                if display_match_score is not None and palate_score is not None:
                    # DISPLAY: Show the actual calculated score (even if 0, it's a real calculation)
                    # MOBILE-OPTIMIZED: Larger text, clearer verdict for in-shop quick glance
                    st.markdown(f"""
<div style="text-align: center; padding: 32px 24px; margin: 20px 0; position: relative;">
    <p style="color: var(--text-secondary); margin: 0 0 12px 0; font-size: 12px; text-transform: uppercase; letter-spacing: 0; font-weight: 600; font-family: var(--font-body);">
        Palate Recommendation Score
    </p>
    <div style="font-size: 4.75rem; margin: 0; font-family: var(--font-display); font-weight: 700; line-height: 1; color: var(--text-primary);">
        {display_match_score:.1f}%
    </div>
    <p style="color: var(--text-primary); margin: 12px 0 0 0; font-size: 1.125rem; font-weight: 600; font-family: var(--font-body);">{palate_score.verdict}</p>
</div>
""", unsafe_allow_html=True)

                    # Breakdown panel: alignment + confidence as two
                    # separate facts. No "x = y" formula — that's the
                    # framing that made users read low-confidence scores
                    # as a regression. Confidence is shown so the user
                    # can read both numbers but isn't asked to multiply
                    # them mentally.
                    n_samples = palate_score.n_samples

                    wines_to_95 = max(0, 10 - n_samples)
                    tip_html = (
                        f' Add <strong style="color: var(--text-primary);">{wines_to_95} more</strong> '
                        f"to reach a more reliable match."
                        if wines_to_95 > 0
                        else ""
                    )

                    # Single-line readout. The earlier version had a
                    # pointless "the headline number is how closely…"
                    # explanation and a separate confidence line. Merged
                    # to one line: the concrete basis (rated-wine count)
                    # plus the nudge. No "confidence" wording — it was
                    # referenced in the nudge while removed everywhere
                    # else, which was inconsistent.
                    st.markdown(f"""<div style="background: var(--card-bg); border: 1px solid var(--card-border); border-radius: var(--radius-card); padding: 1rem 1.5rem; margin: 1.5rem 0;">
<p style="color: var(--text-secondary); font-size: 13px; margin: 0; line-height: 1.6;">Based on <strong style="color: var(--text-primary);">{n_samples} rated wine(s)</strong>.{tip_html}</p>
</div>""", unsafe_allow_html=True)
                else:
                    # LOADING STATE: Show "Calculating..." text instead of 0%
                    st.markdown("""
<div style="text-align: center; padding: 40px 30px; margin: 24px 0;">
    <p style="color: var(--text-secondary); margin: 0 0 16px 0; font-size: 12px; text-transform: uppercase; letter-spacing: 0; font-weight: 600; font-family: var(--font-body);">Palate Recommendation Score</p>
    <div style="font-size: 3rem; margin: 16px 0; font-family: var(--font-display); font-weight: 700; color: var(--text-primary);">
        Calculating...
    </div>
    <p style="color: var(--text-secondary); margin: 16px 0 0 0; font-size: 14px; font-family: var(--font-body);">Analysing your palate profile</p>
</div>
""", unsafe_allow_html=True)

                # Add visual separator
                st.markdown("---")

                # Clean professional presentation: two-column layout.
                st.markdown("### Wine Profile")

                eval_col1, eval_col2 = st.columns(2)

                # LEFT COLUMN: Style, Origin, Vintage
                with eval_col1:
                    st.markdown("**Style & Origin**")
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
                    st.markdown("**Tasting Notes & Verdict**")
                    notes = wine_data.get('notes', 'No tasting notes available')

                    # Display full notes with natural wrapping (no truncation).
                    # No markdown emphasis wrapper. The CSS baseline keeps
                    # all text upright, but plain markdown also makes the
                    # intent explicit at the call site.
                    st.markdown(notes)

                    # Why you'll like it — copy adapts to the alignment
                    # score (the new headline). Thresholds mirror the
                    # engine's STRONG_ALIGNMENT (70) and EXPLORE_ALIGNMENT (55).
                    st.markdown("")  # spacing
                    if display_match_score is not None:
                        if display_match_score >= 70:
                            why_like = (
                                f"**Why you'll like it:** This matches your "
                                f"preferred {wine_color.lower()} style closely."
                            )
                        elif display_match_score >= 55:
                            why_like = (
                                "**Why try it:** Reasonable compatibility "
                                "with your palate, worth exploring."
                            )
                        else:
                            why_like = (
                                f"**Different:** This is a departure from your "
                                f"usual {wine_color.lower()} wines."
                            )
                        st.markdown(why_like)

                st.markdown("---")
            else:
                st.info("Add wines to your collection to see palate match predictions")
                st.markdown("---")

            # 95% PRE-POPULATED "STORE MODE" UI
            st.markdown("### Store Mode - Quick Log")
            st.caption("AI extracted everything - only 3 inputs needed from you!")

            # OPTIMIZED FORM: 3 inputs in one clean row [Score, Price, Like-Toggle]
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                # Score (slider for quick input)
                score_input = st.slider(
                    "Your Score",
                    min_value=1.0,
                    max_value=10.0,
                    value=float(wine_data.get('score', 7.5)),
                    step=0.5,
                    help="How would you rate this wine?"
                )

            with col2:
                # Price - moved from Technical Details for better UX
                price_input = st.number_input(
                    "Price (€)",
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
                    "Did You Like It?",
                    value=liked_default,
                    help="Would you buy this again?"
                )

            # Advanced details in expander (AI-extracted technical data)
            with st.expander("Technical Details & Edit Data (Optional)"):
                st.markdown("#### Flavor Profile (0-10 Scale)")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Acidity", f"{wine_data['acidity']}/10")
                with col2:
                    st.metric("Minerality", f"{wine_data['minerality']}/10")
                with col3:
                    st.metric("Fruitiness", f"{wine_data['fruitiness']}/10")
                with col4:
                    st.metric("Tannin", f"{wine_data['tannin']}/10")
                with col5:
                    st.metric("Body", f"{wine_data['body']}/10")

                st.markdown("---")

                st.markdown("#### Full Technical Specifications")
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
                        st.markdown("**Vintage:** NV")

            # Large, prominent Save button (login required)
            if is_guest:
                st.warning("Log in to save wines to your collection")

            if st.button("SAVE TO MY COLLECTION", type="primary", width="stretch", disabled=is_guest):
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
                        st.error(
                            "Cannot save wine - please fix these issues:\n"
                            + "\n".join(f"• {err}" for err in validation_errors)
                        )
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
                        with st.spinner("Saving wine to Supabase..."):
                            repo_add_wine(get_user_supabase(), row_data)
                        st.success("Wine saved to Supabase.")
                    except Exception as supabase_error:
                        st.error(f"Supabase error while saving wine: {supabase_error}")
                        st.stop()

                    # Save uploaded photo if available
                    photo_bytes = st.session_state.get('uploaded_photo_bytes')
                    photo_name = st.session_state.get('uploaded_photo_name')
                    if photo_bytes and wine_data.get('wine_name'):
                        photo_file = io.BytesIO(photo_bytes)
                        photo_file.name = photo_name or "photo.jpg"
                        saved_path = save_wine_image(photo_file, wine_data['wine_name'])
                        if saved_path:
                            st.info("Photo saved.")

                    # Invalidate the load_wine_data cache so the next
                    # read sees the write. Imported lazily to avoid a
                    # circular import on app.py.
                    from app import clear_wine_data_cache
                    clear_wine_data_cache()

                    st.success(f"Saved {wine_data['wine_name']} to your collection.")
                    st.balloons()

                    # Clear session state to start fresh
                    for key in ['wine_data', 'last_upload', 'uploaded_photo_bytes', 'uploaded_photo_name']:
                        st.session_state.pop(key, None)

                    st.info("Ready for next wine. Add another above.")

                except ValueError as e:
                    st.error(f"Validation error: {str(e)}")
                    st.info("Please check that price is a valid number and liked is true/false")
                except Exception as e:
                    st.error(f"Error saving: {str(e)}")
                    st.info("Check Supabase configuration and RLS permissions")

            else:
                # No data extracted yet
                st.info("Enter a wine name or upload a photo to get started")
