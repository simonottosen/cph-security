import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

/**
 * Simple Accept-Language parser: returns the preferred language code (first token)
 * e.g. "da,en;q=0.9" -> "da"
 */
function preferredLocale(acceptLang: string | null): string {
  if (!acceptLang) return 'en';
  const parts = acceptLang.split(',');
  if (!parts.length) return 'en';
  const first = parts[0].trim().toLowerCase(); // e.g. "da" or "da-dk"
  if (first.startsWith('da')) return 'da';
  return 'en';
}

export function middleware(req: NextRequest) {
  const url = req.nextUrl.clone();

  // Skip API, _next, static assets and files we don't want to rewrite
  const pathname = url.pathname;

  if (
    pathname.startsWith('/api/') ||
    pathname.startsWith('/_next/') ||
    pathname.startsWith('/_static/') ||
    pathname.startsWith('/favicon.ico') ||
    pathname.startsWith('/robots.txt') ||
    pathname.startsWith('/sitemap.xml') ||
    pathname.includes('.')
  ) {
    return;
  }

  // If the path is the site root, do not redirect — keep root as '/'.
  // The root layout will read Accept-Language and set the selector accordingly.
  if (pathname === '/' || pathname === '') {
    return;
  }

  // If path already includes locale prefix (/en/ or /da/), do nothing.
  if (pathname.startsWith('/en') || pathname.startsWith('/da')) {
    return;
  }

  // For any other top-level path like /airports/..., rewrite to include default locale.
  const accept = req.headers.get('accept-language');
  const locale = preferredLocale(accept);
  url.pathname = `/${locale}${pathname}`;
  return NextResponse.rewrite(url);
}

export const config = {
  matcher: [
    /*
     * Match everything except:
     * - API routes (/api/**)
     * - Next internals (_next)
     * - Static files
     */
    '/((?!api|_next|_static|favicon.ico|robots.txt|sitemap.xml).*)',
  ],
};
