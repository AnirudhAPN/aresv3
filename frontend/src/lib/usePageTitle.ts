import { useEffect } from "react";

/** Keeps the browser tab title in sync with the current page, so users
 * navigating with multiple tabs/history entries can tell them apart. */
export function usePageTitle(title: string) {
  useEffect(() => {
    const previous = document.title;
    document.title = `${title} · ARES`;
    return () => {
      document.title = previous;
    };
  }, [title]);
}
