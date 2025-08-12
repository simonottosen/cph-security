"use client";

import React from "react";
import { usePathname, useSearchParams } from "next/navigation";
import { useI18n } from "@/i18n/I18nProvider";

const LOCALES: { code: string; label: string; flag: string }[] = [
  { code: "en", label: "English", flag: "🇬🇧" },
  { code: "da", label: "Dansk", flag: "🇩🇰" },
];

export default function LanguageSwitcher({ initialLocale }: { initialLocale?: string }) {
  const pathname = usePathname() || "/";
  const searchParams = useSearchParams();
  const search = searchParams ? `?${searchParams.toString()}` : "";

  const segs = pathname.split("/").filter(Boolean);
  // derive default from path
  const pathDerived = segs[0] === "da" ? "da" : "en";

  // If a server-provided initialLocale was passed, prefer that for SSR/SSG initial render.
  // useI18n is a client hook and may throw if provider is missing, so call safely.
  let providerLocale: string | undefined = undefined;
  try {
    const i18n = useI18n();
    providerLocale = i18n?.locale;
  } catch {
    // provider not available yet
  }

  const [currentLocale, setCurrentLocale] = React.useState<string>(
    initialLocale ?? providerLocale ?? pathDerived,
  );

  // Keep the switcher's label in sync with the URL (locale segment takes precedence)
  React.useEffect(() => {
    // If the first URL segment is an explicit locale, trust that
    const explicit = segs[0] === "da" || segs[0] === "en" ? segs[0] : undefined;
    const desired = explicit ?? initialLocale ?? providerLocale ?? pathDerived;

    if (desired && desired !== currentLocale) {
      setCurrentLocale(desired);
    }
  }, [segs, initialLocale, providerLocale, pathDerived, currentLocale]);

  // Client-side improvement: when on the root path (no locale prefix),
  // prefer the browser's language preference (navigator.languages) so the selector
  // reflects the visitor's preferred language even if the server used a different header.
  React.useEffect(() => {
    if (typeof window === "undefined") return;
    if (segs.length && segs[0] !== undefined) return; // run only on root (no locale segment)
    if (initialLocale || providerLocale) return;       // server already picked a locale
    try {
      const nav = (navigator.languages && navigator.languages.length) ? navigator.languages : [navigator.language];
      if (nav.some((l: string) => l && l.toLowerCase().startsWith("da"))) {
        setCurrentLocale("da");
      }
    } catch {
      // ignore
    }
  }, [segs]);

  // Remove existing locale prefix if present
  let withoutLocaleSegments = segs;
  if (segs[0] === "en" || segs[0] === "da") {
    withoutLocaleSegments = segs.slice(1);
  }
  const basePath = "/" + withoutLocaleSegments.join("/");

  const makeHref = (newLocale: string) => {
    const path = withoutLocaleSegments.length ? `/${newLocale}${basePath}` : `/${newLocale}/`;
    return `${path}${search}`;
  };

  const [open, setOpen] = React.useState(false);
  const ref = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!ref.current) return;
      if (!ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  const localeMeta = LOCALES.find((l) => l.code === currentLocale) ?? LOCALES[0];

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((s) => !s)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="inline-flex items-center gap-2 rounded-full bg-white/90 dark:bg-gray-900/90 px-3 py-1 ring-1 ring-gray-200 dark:ring-gray-800 shadow-sm hover:shadow-md transition"
      >
        <span className="text-lg">{localeMeta.flag}</span>
        <span className="text-sm font-medium text-gray-700 dark:text-gray-200">{localeMeta.label}</span>
        <svg className="w-3 h-3 text-gray-500" viewBox="0 0 20 20" fill="none" aria-hidden>
          <path d="M6 7l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {open && (
        <div
          role="menu"
          aria-label="Select language"
          className="absolute right-0 mt-2 w-40 rounded-md bg-white dark:bg-gray-900 shadow-lg ring-1 ring-black/5 dark:ring-white/10 z-50 overflow-hidden"
        >
          {LOCALES.map((loc) => (
            <a
              key={loc.code}
              role="menuitem"
              href={makeHref(loc.code)}
              onClick={() => setOpen(false)}
              className="flex items-center gap-2 px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-800 transition text-sm"
            >
              <span className="text-lg">{loc.flag}</span>
              <span className="flex-1">{loc.label}</span>
              {loc.code === currentLocale && (
                <span className="text-xs text-green-600 dark:text-green-400">✓</span>
              )}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
