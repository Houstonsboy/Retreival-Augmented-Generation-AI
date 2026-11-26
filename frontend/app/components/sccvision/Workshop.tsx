import { Check } from "lucide-react";
import type { ReactNode } from "react";

export const STEPS = ["Add evidence", "Device questions", "Review results", "Certificate"];

export const DEVICE_QUESTIONS = [
  {
    key: "source",
    label: "Which device were the screenshots taken on?",
    options: ["My own phone", "Someone else’s phone", "A computer", "Not sure"],
  },
  {
    key: "operation",
    label: "Was the device working normally throughout the period the records cover?",
    options: ["Yes", "No", "Not sure"],
  },
  {
    key: "use",
    label: "Who usually uses this device?",
    options: ["Only me", "Me and others", "Someone else", "Not sure"],
  },
] as const;

export type DeviceAnswers = Partial<Record<(typeof DEVICE_QUESTIONS)[number]["key"], string>>;

export function Choices({ name, legend, value, options, onChange }: {
  name: string;
  legend: string;
  value?: string;
  options: readonly { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <fieldset className="scc-card scc-question">
      <legend className="sr-only">{legend}</legend>
      <p className="scc-label" aria-hidden="true">{legend}</p>
      <div className="scc-options">
        {options.map((option) => (
          <label key={option.value}>
            <input type="radio" name={name} value={option.value} checked={option.value === value}
              onChange={() => onChange(option.value)} />
            {option.label}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export function StepHeading({ step, title, children }: { step: number; title: string; children: ReactNode }) {
  return (
    <>
      <nav aria-label="Analysis progress">
        <ol className="scc-steps">
          {STEPS.map((label, index) => (
            <li key={label} data-complete={index < step} aria-current={index === step ? "step" : undefined}>
              <span aria-hidden="true">{index < step ? <Check size={12} /> : index + 1}</span>{label}
            </li>
          ))}
        </ol>
      </nav>
      <div className="scc-heading">
        <p className="scc-kicker">Step {step + 1} of {STEPS.length}{step === 1 && " · About 1 minute"}</p>
        <h1 id="sccvision-heading" tabIndex={-1}>{title}</h1>
        <p className="scc-intro">{children}</p>
      </div>
    </>
  );
}
