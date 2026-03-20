"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Link } from "lucide-react";

interface UrlInputProps {
  onUrlSubmit: (url: string) => void;
  disabled?: boolean;
}

export function UrlInput({ onUrlSubmit, disabled }: UrlInputProps) {
  const [url, setUrl] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (url.trim()) {
      onUrlSubmit(url.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="flex gap-3">
        <div className="relative flex-1">
          <Link className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="url"
            placeholder="https://youtube.com/watch?v=... or https://vimeo.com/..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className="pl-10"
            disabled={disabled}
          />
        </div>
        <Button type="submit" disabled={!url.trim() || disabled}>
          Process URL
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Supports YouTube, Vimeo, and most video platforms
      </p>
    </form>
  );
}
