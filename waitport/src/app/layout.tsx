import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { headers } from "next/headers";
import { I18nProvider } from "@/i18n/I18nProvider";
import LanguageSwitcher from "@/components/LanguageSwitcher";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "Waitport - Real-time & Predicted Airport Security Queues",
  description:
    "Check live and predicted security queue wait times at major European airports. Plan your trip effectively with Waitport's real-time data and future estimates.",
};

export const dynamic = "force-dynamic";

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Try to read the locale set by Next or middleware. Fallback to Accept-Language.
  const hdrs = await headers();
  const nextLocale = hdrs.get?.("x-nextjs-locale") ?? undefined;
  const acceptLang = (hdrs.get?.("accept-language") ?? "") as string;

  // Parse Accept-Language by taking the first token (highest priority).
  // e.g. "en-US,en;q=0.9,da;q=0.8" -> "en-US" -> English
  const parsePreferredLocale = (al: string) => {
    if (!al) return "en";
    const first = al.split(",")[0].trim().toLowerCase();
    if (first.startsWith("da")) return "da";
    return "en";
  };

  const locale = nextLocale ?? parsePreferredLocale(acceptLang);

  // Load messages for the resolved locale
  let messages: Record<string, any> = {};
  try {
    const mod = await import(`../locales/${locale}.json`);
    messages = (mod && (mod.default ?? mod)) as Record<string, any>;
  } catch (err) {
    const mod = await import("../locales/en.json");
    messages = (mod && (mod.default ?? mod)) as Record<string, any>;
  }

  return (
    <html lang={locale}>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <I18nProvider locale={locale} messages={messages}>
          <div className="absolute top-4 right-4 z-50">
            <LanguageSwitcher initialLocale={locale} />
          </div>
          {children}
        </I18nProvider>
      </body>
    </html>
  );
}
