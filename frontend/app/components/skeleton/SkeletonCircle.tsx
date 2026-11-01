import { SkeletonBar } from "./SkeletonBar";

interface SkeletonCircleProps {
  size?: string | number;
}

export function SkeletonCircle({ size = "2rem" }: SkeletonCircleProps) {
  return <SkeletonBar width={size} height={size} radius="50%" />;
}
