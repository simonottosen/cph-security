'use client';

import React, { createContext, useContext, ReactNode, useMemo } from 'react';

type Messages = Record<string, string>;

type I18nContextValue = {
  locale: string;
  messages: Messages;
  t: (key: string, vars?: Record<string | number, string | number>) => string;
};

const I18nContext = createContext<I18nContextValue | null>(null);

export function I18nProvider({
  locale,
  messages,
  children,
}: {
  locale: string;
  messages: Messages;
  children: ReactNode;
}) {
  const t = useMemo(() => {
    return (key: string, vars?: Record<string | number, string | number>) => {
      const raw = messages?.[key] ?? key;
      if (!vars) return raw;
      return Object.entries(vars).reduce((s, [k, v]) => {
        return s.replace(new RegExp(`\\{${k}\\}`, 'g'), String(v));
      }, raw);
    };
  }, [messages]);

  const value: I18nContextValue = {
    locale,
    messages,
    t,
  };

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    throw new Error('useI18n must be used within an I18nProvider');
  }
  return ctx;
}
