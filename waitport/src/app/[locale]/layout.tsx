import type { Metadata } from "next";
import localFont from "next/font/local";
import "../globals.css";
import { I18nProvider } from "@/i18n/I18nProvider";
import LanguageSwitcher from "@/components/LanguageSwitcher";

const geistSans = localFont({
  src: "../fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "../fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const dynamic = "force-dynamic";

export default async function LocaleLayout({
  children,
  params,
}: Readonly<{
  children: React.ReactNode;
  params: { locale?: string };
}>) {
  // params in App Router can be a promise — await it before using properties
  const resolvedParams = (await params) as { locale?: string } | undefined;
  const locale = resolvedParams?.locale ?? "en";

  // Load messages dynamically per-request
  let messages: Record<string, any> = {};
  try {
    // relative to this file: ../../locales/{locale}.json -> src/locales
    // use dynamic import so Next can include the JSON as an asset
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = await import(`../../locales/${locale}.json`);
    messages = (mod && (mod.default ?? mod)) as Record<string, any>;
  } catch (err) {
    // fallback to English
    const mod = await import("../../locales/en.json");
    messages = (mod && (mod.default ?? mod)) as Record<string, any>;
  }

  return (
    <html lang={locale}>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        {/* Language selector (top-right) */}
        <div className="absolute top-4 right-4 z-50">
          <LanguageSwitcher />
        </div>
        {/* I18nProvider is a client component; it's safe to render here */}
        <I18nProvider locale={locale} messages={messages}>
          {children}
        </I18nProvider>
      </body>
    </html>
  );
}
