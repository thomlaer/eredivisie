import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Eredivisie voorspeller",
  description: "Datagedreven Eredivisie-voorspellingen, standen en kampioenskansen.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="nl">
      <body>{children}</body>
    </html>
  );
}
