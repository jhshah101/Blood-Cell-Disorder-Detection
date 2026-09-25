import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Activity, Zap } from "lucide-react";

export interface ClassificationResult {
  cellType: string;
  /** Softmax score of the predicted class in [0, 1]. */
  confidence: number;
  /** Softmax score per class in [0, 1]; keys are the checkpoint's class names. */
  distribution: Record<string, number>;
  latencyMs?: number;
  model?: string;
  disclaimer?: string;
}

interface ResultsCardProps {
  result: ClassificationResult;
}

const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

const ResultsCard = ({ result }: ResultsCardProps) => {
  const { cellType, confidence, distribution, latencyMs, model, disclaimer } = result;
  const rows = Object.entries(distribution).sort((a, b) => b[1] - a[1]);

  return (
    <Card className="p-8 bg-gradient-card shadow-medical border-0 relative overflow-hidden animate-scale-in">
      <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-medical opacity-5 rounded-full -translate-y-16 translate-x-16" />

      <div className="space-y-8 relative z-10">
        <div className="text-center space-y-6">
          <div className="flex flex-col items-center gap-4">
            <div className="p-4 bg-gradient-medical rounded-2xl shadow-medical">
              <Zap className="h-6 w-6 text-white" />
            </div>
            <div className="space-y-3">
              <h3 className="text-3xl font-bold text-foreground">{cellType}</h3>
              <div className="flex flex-wrap items-center gap-3 justify-center">
                <Badge variant="secondary" className="bg-gradient-medical text-white px-4 py-2 text-base font-semibold rounded-full">
                  softmax score {pct(confidence)}
                </Badge>
                {typeof latencyMs === "number" && (
                  <span className="text-sm text-muted-foreground">inference {latencyMs.toFixed(1)} ms</span>
                )}
                {model && <span className="text-sm text-muted-foreground">model: {model}</span>}
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="flex items-center gap-3 justify-center">
            <Activity className="h-5 w-5 text-primary" />
            <h4 className="text-lg font-bold text-foreground">Class scores</h4>
          </div>

          <div className="space-y-4">
            {rows.map(([type, score], index) => (
              <div
                key={type}
                className="space-y-2 animate-fade-in p-3 rounded-lg hover:bg-background/50 transition-all duration-300"
                style={{ animationDelay: `${index * 80}ms` }}
              >
                <div className="flex justify-between items-center text-sm">
                  <span className="text-foreground font-semibold flex items-center gap-2">
                    {type === cellType && <span className="w-2 h-2 bg-primary rounded-full" aria-hidden="true" />}
                    {type}
                  </span>
                  <span className="text-lg font-bold text-foreground tabular-nums">{pct(score)}</span>
                </div>
                <div className="w-full bg-secondary rounded-full h-3 overflow-hidden" role="progressbar" aria-valuenow={Math.round(100 * score)} aria-valuemin={0} aria-valuemax={100} aria-label={`${type} score`}>
                  <div
                    className={`h-3 rounded-full transition-all duration-700 ease-out ${type === cellType ? "bg-gradient-medical shadow-glow" : "bg-muted-foreground/40"}`}
                    style={{ width: `${Math.max(0, Math.min(100, 100 * score))}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <p className="text-xs text-muted-foreground text-center leading-relaxed">
          {disclaimer ??
            "Scores are softmax outputs of an image classifier and are not calibrated clinical probabilities."}
        </p>
      </div>
    </Card>
  );
};

export default ResultsCard;
