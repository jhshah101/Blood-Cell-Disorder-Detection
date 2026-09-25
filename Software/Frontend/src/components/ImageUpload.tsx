import { useCallback, useRef, useState } from "react";
import { Image as ImageIcon, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface ImageUploadProps {
  onImageUpload: (file: File) => void;
  onInvalidFile?: (reason: string) => void;
  onClassify: () => void;
  uploadedImage: string | null;
  isClassifying: boolean;
  disabled?: boolean;
  maxSizeMb?: number;
}

const ACCEPTED = ["image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"];

const ImageUpload = ({
  onImageUpload,
  onInvalidFile,
  onClassify,
  uploadedImage,
  isClassifying,
  disabled = false,
  maxSizeMb = 20,
}: ImageUploadProps) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        onInvalidFile?.(`"${file.name}" is not an image.`);
        return;
      }
      if (!ACCEPTED.includes(file.type)) {
        onInvalidFile?.(`Unsupported image type ${file.type}. Use JPEG, PNG, BMP, TIFF or WebP.`);
        return;
      }
      if (file.size > maxSizeMb * 1024 * 1024) {
        onInvalidFile?.(`"${file.name}" is larger than ${maxSizeMb} MB.`);
        return;
      }
      onImageUpload(file);
    },
    [onImageUpload, onInvalidFile, maxSizeMb],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      accept(e.dataTransfer.files?.[0]);
    },
    [accept],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      accept(e.target.files?.[0]);
      e.target.value = ""; // allow re-selecting the same file
    },
    [accept],
  );

  return (
    <div className="space-y-6">
      <Card
        role="button"
        tabIndex={0}
        aria-label="Upload a microscope image"
        className={`p-8 border-2 border-dashed transition-all duration-500 cursor-pointer bg-gradient-card shadow-card hover:shadow-hover relative overflow-hidden ${
          isDragOver ? "border-primary bg-primary/10" : "border-border hover:border-primary/50"
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        <input ref={inputRef} type="file" accept={ACCEPTED.join(",")} onChange={handleFileInput} className="hidden" />

        <div className="flex flex-col items-center text-center space-y-4 relative z-10">
          {uploadedImage ? (
            <div className="space-y-4 animate-scale-in">
              <div className="relative w-40 h-40 mx-auto rounded-xl overflow-hidden shadow-medical border-2 border-primary/20">
                <img src={uploadedImage} alt="Uploaded microscope image" className="w-full h-full object-cover" />
              </div>
              <div className="space-y-2">
                <p className="text-base font-semibold text-foreground flex items-center gap-2 justify-center">
                  <span className="w-2 h-2 bg-green-500 rounded-full" aria-hidden="true" />
                  Image ready for analysis
                </p>
                <p className="text-sm text-muted-foreground">Click or drop to replace it</p>
              </div>
            </div>
          ) : (
            <>
              <div className="p-6 bg-gradient-medical rounded-full shadow-medical">
                <Upload className="h-10 w-10 text-white" />
              </div>
              <div className="space-y-3">
                <h3 className="text-xl font-bold text-foreground">Upload a single-cell image</h3>
                <p className="text-base text-muted-foreground max-w-md leading-relaxed">
                  Drag and drop a cropped white blood cell image (as in Raabin-WBC), or click to browse. The image is
                  resized to the model's input size on the server.
                </p>
                <div className="flex flex-wrap justify-center gap-2 text-xs text-muted-foreground">
                  {["JPG", "PNG", "BMP", "TIFF", "WebP"].map((t) => (
                    <span key={t} className="px-2 py-1 bg-background/50 rounded-full">
                      {t}
                    </span>
                  ))}
                  <span className="px-2 py-1 bg-background/50 rounded-full">≤ {maxSizeMb} MB</span>
                </div>
              </div>
            </>
          )}
        </div>
      </Card>

      {uploadedImage && (
        <div className="flex flex-col items-center gap-2 animate-fade-in">
          <Button
            onClick={onClassify}
            disabled={isClassifying || disabled}
            className="bg-gradient-medical hover:shadow-hover transition-all duration-300 px-10 py-4 text-lg font-bold rounded-xl"
          >
            {isClassifying ? (
              <>
                <span className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent mr-3" aria-hidden="true" />
                Classifying…
              </>
            ) : (
              <>
                <ImageIcon className="h-5 w-5 mr-3" />
                Classify cell
              </>
            )}
          </Button>
          {disabled && <p className="text-sm text-muted-foreground">The backend is not reachable, so classification is disabled.</p>}
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
