import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Layers, Palette, Scale, ShieldCheck } from "lucide-react";
import type { ModelInfo } from "@/lib/api";

interface InfoSectionProps {
  info: ModelInfo | null;
}

const InfoSection = ({ info }: InfoSectionProps) => {
  const features = [
    {
      icon: <Layers className="h-5 w-5" />,
      title: "Hybrid CNN–Transformer",
      description: "ResNet-18 structural tokens refined by Efficient Channel Attention and encoded by a Transformer.",
      active: info?.use_eca ?? true,
    },
    {
      icon: <Palette className="h-5 w-5" />,
      title: "Colour-feature fusion",
      description: "Per-region colour statistics in RGB, HSV and Lab concatenated to every token to capture staining cues.",
      active: info?.use_color_features ?? true,
    },
    {
      icon: <Scale className="h-5 w-5" />,
      title: "Adaptive loss reweighting",
      description: "Class weights re-estimated every epoch from class difficulty and confidence, rescaled to mean one.",
      active: info ? info.loss === "alr" : true,
    },
    {
      icon: <ShieldCheck className="h-5 w-5" />,
      title: "Leakage-safe protocol",
      description: "Checkpoint chosen on a validation split; the test set is evaluated once, after training.",
      active: true,
    },
  ];

  // Approximate adult reference ranges of the leukocyte differential (for orientation only).
  const cellTypes = [
    { name: "Neutrophils", description: "Most abundant granulocyte; first responders to bacterial infection", percentage: "40–70%" },
    { name: "Lymphocytes", description: "B, T and NK cells of adaptive immunity", percentage: "20–40%" },
    { name: "Monocytes", description: "Large mononuclear cells that differentiate into macrophages", percentage: "2–8%" },
    { name: "Eosinophils", description: "Granulocytes involved in parasitic infection and allergy", percentage: "1–4%" },
    { name: "Basophils", description: "Rarest granulocyte; histamine-containing granules", percentage: "0.5–1%" },
  ];

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {features.map((feature, index) => (
          <Card
            key={feature.title}
            className={`p-6 bg-gradient-card shadow-card transition-all duration-300 animate-fade-in border-0 ${feature.active ? "" : "opacity-60"}`}
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <div className="space-y-3">
              <div className="p-2 bg-primary/10 rounded-lg w-fit">
                <div className="text-primary">{feature.icon}</div>
              </div>
              <div>
                <h3 className="font-semibold text-foreground">
                  {feature.title}
                  {!feature.active && <span className="ml-2 text-xs text-muted-foreground">(disabled in this checkpoint)</span>}
                </h3>
                <p className="text-sm text-muted-foreground mt-1">{feature.description}</p>
              </div>
            </div>
          </Card>
        ))}
      </div>

      <Card className="p-6 bg-gradient-card shadow-medical border-0 animate-fade-in">
        <div className="space-y-6">
          <div className="text-center space-y-2">
            <h3 className="text-xl font-bold text-foreground">White blood cell types</h3>
            <p className="text-muted-foreground">
              The five classes of the Raabin-WBC dataset with approximate adult differential ranges (orientation only).
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {cellTypes.map((cell) => (
              <div key={cell.name} className="p-4 rounded-lg bg-background/50 border border-border/50 hover:border-primary/30 transition-all duration-300">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="font-semibold text-foreground">{cell.name}</h4>
                    <Badge variant="secondary" className="text-xs">
                      {cell.percentage}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">{cell.description}</p>
                </div>
              </div>
            ))}
          </div>

          <p className="text-xs text-muted-foreground text-center leading-relaxed">
            Basophils, eosinophils and monocytes are the minority classes of the training set; the model's minority-class
            F1 and the minority-to-majority F1 balance ratio are the metrics the project optimises and reports.
          </p>
        </div>
      </Card>
    </div>
  );
};

export default InfoSection;
