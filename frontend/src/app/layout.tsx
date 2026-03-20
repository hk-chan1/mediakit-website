import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "MediaKit - Video to Piano Sheet Music",
  description:
    "Upload a video or paste a URL and get AI-generated piano sheet music in seconds. Extract melody, chords, and rhythm automatically.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen flex flex-col">
          <header className="border-b">
            <div className="container mx-auto px-4 py-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                  <span className="text-white font-bold text-sm">MK</span>
                </div>
                <span className="font-bold text-xl">MediaKit</span>
              </div>
              <nav className="text-sm text-muted-foreground">
                <a href="#how-it-works" className="hover:text-foreground transition-colors">
                  How it works
                </a>
              </nav>
            </div>
          </header>
          <main className="flex-1">{children}</main>
          <footer className="border-t py-6">
            <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
              MediaKit &mdash; AI-powered video to sheet music conversion
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
