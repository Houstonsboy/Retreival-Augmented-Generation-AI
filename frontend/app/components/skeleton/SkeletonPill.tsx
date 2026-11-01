import { SkeletonBar } from "./SkeletonBar";

interface SkeletonPillProps {
  width?: string | number;
  height?: string | number;
}

export function SkeletonPill({ width = "80px", height = "1.5rem" }: SkeletonPillProps) {
  return <SkeletonBar width={width} height={height} radius="9999px" />;
}
