-- Decant — RLS policies for the wines table (Phase 2)
--
-- Model: one shared household cellar identified by CELLAR_ID.
-- - Anyone (anon role) can SELECT wines tagged with the shared CELLAR_ID
--   so guests browsing the public URL see the gallery and palate maps.
-- - Only authenticated users can INSERT/UPDATE/DELETE, also scoped to
--   the shared CELLAR_ID. There's no per-user attribution; both
--   household members write into the same cellar.
--
-- Replace 'YOUR_CELLAR_UUID_HERE' with the same UUID you set in
-- secrets.toml under CELLAR_ID. Run this script in the Supabase SQL
-- editor.
--
-- Idempotency: every DROP/CREATE is safe to re-run; existing policies
-- with matching names are replaced.

-- Make sure RLS is on. Without this, the policies below have no effect
-- and the table is wide open to the anon key.
ALTER TABLE public.wines ENABLE ROW LEVEL SECURITY;

-- 1. Anonymous read of the shared cellar.
--    Friends browsing the URL hit this path.
DROP POLICY IF EXISTS "anon_read_shared_cellar" ON public.wines;
CREATE POLICY "anon_read_shared_cellar"
    ON public.wines
    FOR SELECT
    TO anon
    USING (cellar_id = 'YOUR_CELLAR_UUID_HERE'::uuid);

-- 2. Authenticated read of the shared cellar.
--    Household members see the same rows; the policy is separate so we
--    can later restrict anon access (e.g. hide a column) without
--    breaking authenticated reads.
DROP POLICY IF EXISTS "authed_read_shared_cellar" ON public.wines;
CREATE POLICY "authed_read_shared_cellar"
    ON public.wines
    FOR SELECT
    TO authenticated
    USING (cellar_id = 'YOUR_CELLAR_UUID_HERE'::uuid);

-- 3. Authenticated insert into the shared cellar.
--    WITH CHECK enforces that the inserted row has the correct
--    cellar_id; a logged-in user can't write to other cellars even if
--    they craft the payload manually.
DROP POLICY IF EXISTS "authed_insert_shared_cellar" ON public.wines;
CREATE POLICY "authed_insert_shared_cellar"
    ON public.wines
    FOR INSERT
    TO authenticated
    WITH CHECK (cellar_id = 'YOUR_CELLAR_UUID_HERE'::uuid);

-- 4. Authenticated update of the shared cellar.
DROP POLICY IF EXISTS "authed_update_shared_cellar" ON public.wines;
CREATE POLICY "authed_update_shared_cellar"
    ON public.wines
    FOR UPDATE
    TO authenticated
    USING (cellar_id = 'YOUR_CELLAR_UUID_HERE'::uuid)
    WITH CHECK (cellar_id = 'YOUR_CELLAR_UUID_HERE'::uuid);

-- 5. Authenticated delete of the shared cellar.
DROP POLICY IF EXISTS "authed_delete_shared_cellar" ON public.wines;
CREATE POLICY "authed_delete_shared_cellar"
    ON public.wines
    FOR DELETE
    TO authenticated
    USING (cellar_id = 'YOUR_CELLAR_UUID_HERE'::uuid);

-- Verification queries.
-- Run after applying the policies to confirm they're in place.
-- (Comment these out before running in production if you want.)
SELECT
    schemaname,
    tablename,
    policyname,
    cmd,
    roles
FROM pg_policies
WHERE tablename = 'wines'
ORDER BY policyname;

-- Smoke test: an anon-key client should be able to read.
-- This runs as the SQL editor session, not anon, so it's not a real
-- check — for a real check, hit the REST API with the anon key from
-- your local machine:
--
--   curl '<SUPABASE_URL>/rest/v1/wines?select=wine_name&limit=3' \
--     -H "apikey: <SUPABASE_ANON_KEY>" \
--     -H "Authorization: Bearer <SUPABASE_ANON_KEY>"
--
-- Should return rows with the matching cellar_id.
