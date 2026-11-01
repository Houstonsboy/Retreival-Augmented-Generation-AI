import { SkeletonBar } from "./SkeletonBar";

interface SkeletonCircleProps { size?: string | number }

export function SkeletonCircle({ size = "24px" }: SkeletonCircleProps) {
  return <SkeletonBar width={size} height={size} radius="50%" />;
}
