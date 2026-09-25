import { Microscope } from "lucide-react";

export type BackendStatus = "checking" | "online" | "offline";

interface HeaderProps {
  status: BackendStatus;
}

const STATUS_LABEL: Record<BackendStatus, { text: string; dot: string }> = {
  checking: { text: "Checking backend…", dot: "bg-yellow-500" },
  online: { text: "Backend online", dot: "bg-green-500" },
  offline: { text: "Backend offline", dot: "bg-red-500" },
};

const Header = ({ status }: HeaderProps) => {
  const s = STATUS_LABEL[status];
  return (
    <header className="w-full bg-gradient-subtle border-b border-border/50 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-accent/5 opacity-50" />

      <div className="container mx-auto px-4 py-8 relative">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-medical rounded-xl shadow-medical">
              <Microscope className="h-7 w-7 text-white" />
            </div>
            <div className="space-y-1">
              <h1 className="text-3xl font-bold text-foreground">White Blood Cell Classifier</h1>
              <p className="text-muted-foreground text-sm">ViT-ECA-CF with adaptive loss reweighting — research prototype</p>
            </div>
          </div>

          <div className="hidden md:flex items-center gap-2 text-sm text-muted-foreground" role="status" aria-live="polite">
            <span className={`w-2 h-2 rounded-full ${s.dot}`} aria-hidden="true" />
            {s.text}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
