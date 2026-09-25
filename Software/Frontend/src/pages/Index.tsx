import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Header from "@/components/Header";
import ImageUpload from "@/components/ImageUpload";
import ResultsCard, { ClassificationResult } from "@/components/ResultsCard";
import InfoSection from "@/components/InfoSection";
import StatsOverlay from "@/components/StatsOverlay";
import { useToast } from "@/hooks/use-toast";
import { ApiError, classifyImage, fetchHealth, fetchModelInfo } from "@/lib/api";

const Index = () => {
  const { toast } = useToast();
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const health = useQuery({ queryKey: ["health"], queryFn: fetchHealth, refetchInterval: 15_000, retry: 1 });
  const modelInfo = useQuery({
    queryKey: ["model-info"],
    queryFn: fetchModelInfo,
    enabled: health.data?.model_loaded === true,
    staleTime: Infinity,
  });

  // Release object URLs when the preview changes or the page unmounts.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleImageUpload = useCallback((file: File) => {
    setUploadedFile(file);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(file);
    });
    setResult(null);
  }, []);

  const handleInvalidFile = useCallback(
    (reason: string) => toast({ title: "File not accepted", description: reason, variant: "destructive" }),
    [toast],
  );

  const handleClassify = useCallback(async () => {
    if (!uploadedFile || isClassifying) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setIsClassifying(true);
    try {
      const data = await classifyImage(uploadedFile, controller.signal);
      setResult({
        cellType: data.predicted_class,
        confidence: data.confidence,
        distribution: data.probabilities,
        latencyMs: data.latency_ms,
        model: data.model,
        disclaimer: data.disclaimer,
      });
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Unexpected error while classifying the image.";
      toast({ title: "Classification failed", description: message, variant: "destructive" });
    } finally {
      setIsClassifying(false);
    }
  }, [uploadedFile, isClassifying, toast]);

  const status = health.isLoading ? "checking" : health.data?.model_loaded ? "online" : "offline";

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <Header status={status} />

      <main className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto space-y-12">
          <div className="text-center space-y-6 animate-fade-in">
            <h2 className="text-2xl md:text-3xl font-bold text-foreground leading-tight">
              White blood cell classification with a hybrid CNN&ndash;Transformer
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Upload a single-cell microscope crop to obtain the five-class prediction and the softmax scores of
              the ViT-ECA-CF model. This is a research prototype trained on the public Raabin-WBC dataset; it is
              not a diagnostic device and its scores are not calibrated clinical probabilities.
            </p>
          </div>

          <div className="animate-fade-in" style={{ animationDelay: "200ms" }}>
            <StatsOverlay info={modelInfo.data ?? null} loading={modelInfo.isLoading || health.isLoading} />
          </div>

          <div className="max-w-2xl mx-auto">
            <ImageUpload
              onImageUpload={handleImageUpload}
              onInvalidFile={handleInvalidFile}
              onClassify={handleClassify}
              uploadedImage={previewUrl}
              isClassifying={isClassifying}
              disabled={status !== "online"}
            />

            {result && (
              <div className="mt-8 animate-scale-in">
                <ResultsCard result={result} />
              </div>
            )}
          </div>

          <div className="animate-fade-in" style={{ animationDelay: "400ms" }}>
            <InfoSection info={modelInfo.data ?? null} />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
