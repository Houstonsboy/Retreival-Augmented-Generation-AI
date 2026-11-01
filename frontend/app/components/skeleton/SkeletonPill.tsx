import { SkeletonBar } from "./SkeletonBar";

interface SkeletonPillProps {
  width?:  string | number;
  height?: string | number;
}

export function SkeletonPill({ width = "80px", height = "20px" }: SkeletonPillProps) {
  return <SkeletonBar width={width} height={height} radius="9999px" />;
}
