import { I18nProvider } from "@/i18n/I18nProvider";

export const dynamic = "force-dynamic";

export default async function LocaleLayout({
  children,
  params,
}: Readonly<{
  children: React.ReactNode;
  params: { locale?: string };
}>) {
  const resolvedParams = (await params) as { locale?: string } | undefined;
  const locale = resolvedParams?.locale ?? "en";

  let messages: Record<string, any> = {};
  try {
    const mod = await import(`../../locales/${locale}.json`);
    messages = (mod && (mod.default ?? mod)) as Record<string, any>;
  } catch {
    const mod = await import("../../locales/en.json");
    messages = (mod && (mod.default ?? mod)) as Record<string, any>;
  }

  return (
    <I18nProvider locale={locale} messages={messages}>
      {children}
    </I18nProvider>
  );
}
