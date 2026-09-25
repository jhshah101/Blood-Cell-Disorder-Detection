import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Cpu, Layers, Ruler, Target } from "lucide-react";
import type { ModelInfo } from "@/lib/api";

interface StatsOverlayProps {
  info: ModelInfo | null;
  loading?: boolean;
}

/**
 * Facts about the *loaded checkpoint*, read from the backend's /model-info
 * endpoint.  Nothing here is hard-coded marketing copy: if the backend is
 * offline the tiles say so.
 */
const StatsOverlay = ({ info, loading = false }: StatsOverlayProps) => {
  const valF1 = info?.checkpoint?.validation?.macro_f1;
  const stats = [
    {
      icon: <Cpu className="h-6 w-6" />,
      value: info ? info.backbone : "—",
      label: "Backbone",
      color: "text-blue-600",
    },
    {
      icon: <Layers className="h-6 w-6" />,
      value: info ? String(info.classes.length) : "—",
      label: "Classes",
      color: "text-purple-600",
    },
    {
      icon: <Target className="h-6 w-6" />,
      value: typeof valF1 === "number" ? `${(100 * valF1).toFixed(1)}%` : "—",
      label: "Validation macro-F1",
      color: "text-green-600",
    },
    {
      icon: <Ruler className="h-6 w-6" />,
      value: info ? `${info.img_size} px` : "—",
      label: "Input size",
      color: "text-orange-600",
    },
  ];

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <Card
            key={stat.label}
            className="p-4 text-center bg-background/80 backdrop-blur-sm border-border/50 shadow-card transition-all duration-300 animate-fade-in"
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <div className="space-y-2">
              <div className={`${stat.color} mx-auto w-fit`}>{stat.icon}</div>
              {loading ? (
                <Skeleton className="h-6 w-16 mx-auto" />
              ) : (
                <div className="text-xl font-bold text-foreground truncate" title={stat.value}>
                  {stat.value}
                </div>
              )}
              <div className="text-xs text-muted-foreground">{stat.label}</div>
            </div>
          </Card>
        ))}
      </div>
      <p className="text-xs text-muted-foreground text-center">
        {info
          ? `Checkpoint selected on the validation split (epoch ${info.checkpoint.epoch ?? "?"}); ${info.loss} loss, ${info.augmentation === "none" ? "no augmentation" : "augmented"}.`
          : "Backend offline or no checkpoint loaded — model facts are shown once the server is reachable."}
      </p>
    </div>
  );
};

export default StatsOverlay;
