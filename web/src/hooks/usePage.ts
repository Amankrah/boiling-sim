// Query-param routing for the three dashboard pages.
//
// We deliberately don't pull in react-router-dom for this -- the app
// has exactly three routes and the share-link mechanism already
// round-trips scene params via URLSearchParams, so a 40-line hook
// keeps the bundle clean.
//
// Tab switches use `history.replaceState` (not push) so the browser
// back button doesn't build up a long history of tab clicks; F5
// preserves the tab.

import { useCallback, useEffect, useState } from "react";

export type Page = "live" | "config" | "results";

const VALID_PAGES: readonly Page[] = ["live", "config", "results"] as const;
const DEFAULT_PAGE: Page = "live";

function readPageFromUrl(): Page {
  if (typeof window === "undefined") return DEFAULT_PAGE;
  const raw = new URLSearchParams(window.location.search).get("page");
  if (raw && (VALID_PAGES as readonly string[]).includes(raw)) {
    return raw as Page;
  }
  return DEFAULT_PAGE;
}

/** Rewrite `?page=...` in the URL without clobbering other search
 *  params (the share-link scheme writes `hf`, `mat`, etc.). */
function writePageToUrl(page: Page): void {
  if (typeof window === "undefined") return;
  const params = new URLSearchParams(window.location.search);
  params.set("page", page);
  const url = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState(null, "", url);
}

export interface UsePageReturn {
  page: Page;
  setPage: (page: Page) => void;
}

export function usePage(): UsePageReturn {
  const [page, setPageState] = useState<Page>(() => readPageFromUrl());

  const setPage = useCallback((next: Page) => {
    setPageState(next);
    writePageToUrl(next);
  }, []);

  // Keep state in sync with back/forward navigation -- popstate fires
  // when the user walks the browser history even if we only used
  // replaceState, because another SPA on the same origin might push.
  useEffect(() => {
    const onPop = () => setPageState(readPageFromUrl());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  return { page, setPage };
}
