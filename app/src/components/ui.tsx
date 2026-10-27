// Shared UI primitives: buttons, badges, spinner, modal, tabs, empty state.
import { useEffect, type ReactNode, type ButtonHTMLAttributes } from "react";
import { X, type LucideIcon } from "lucide-react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const VARIANT: Record<Variant, string> = {
  primary:
    "bg-accent-purple text-surface-0 hover:bg-accent-purple/90 font-semibold shadow-sm shadow-accent-purple/20",
  secondary:
    "bg-surface-3 text-text-primary border border-surface-4 hover:bg-surface-4 hover:border-text-muted/40",
  ghost: "text-text-secondary hover:text-text-primary hover:bg-surface-3",
  danger: "text-p1-light hover:bg-p1/15",
};

const SIZE: Record<Size, string> = {
  sm: "h-7 px-2.5 text-xs gap-1.5",
  md: "h-9 px-3.5 text-sm gap-2",
  lg: "h-11 px-5 text-sm gap-2",
};

export function Button({
  variant = "secondary",
  size = "md",
  icon: Icon,
  loading,
  className = "",
  children,
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: Size;
  icon?: LucideIcon;
  loading?: boolean;
}) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={`inline-flex items-center justify-center rounded-lg font-medium whitespace-nowrap
        transition-colors duration-150 cursor-pointer select-none
        disabled:opacity-50 disabled:cursor-not-allowed
        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-purple/60
        ${VARIANT[variant]} ${SIZE[size]} ${className}`}
    >
      {loading ? <Spinner size={14} /> : Icon && <Icon size={size === "sm" ? 14 : 16} strokeWidth={2} />}
      {children}
    </button>
  );
}

export function IconButton({
  icon: Icon,
  label,
  active,
  size = 32,
  className = "",
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  icon: LucideIcon;
  label: string;
  active?: boolean;
  size?: number;
}) {
  return (
    <button
      {...rest}
      aria-label={label}
      title={rest.title ?? label}
      style={{ width: size, height: size }}
      className={`inline-flex items-center justify-center rounded-md transition-colors cursor-pointer shrink-0
        disabled:opacity-40 disabled:cursor-not-allowed
        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-purple/60
        ${active ? "bg-accent-purple/15 text-accent-purple" : "text-text-secondary hover:text-text-primary hover:bg-surface-3"}
        ${className}`}
    >
      <Icon size={Math.round(size * 0.5)} strokeWidth={2} />
    </button>
  );
}

export function Badge({
  children,
  tone = "neutral",
  className = "",
}: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "green" | "p1" | "p2" | "gold";
  className?: string;
}) {
  const tones = {
    neutral: "bg-surface-3 text-text-secondary",
    accent: "bg-accent-purple/15 text-accent-purple",
    green: "bg-accent-green/15 text-accent-green",
    gold: "bg-accent-gold/15 text-accent-gold",
    p1: "bg-p1/15 text-p1-light",
    p2: "bg-p2/15 text-p2-light",
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-semibold leading-none tabular-nums ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

export function Spinner({ size = 20, className = "" }: { size?: number; className?: string }) {
  return (
    <span
      className={`inline-block rounded-full border-2 animate-spin shrink-0 ${className}`}
      style={{ width: size, height: size, borderColor: "currentColor", borderTopColor: "transparent" }}
      aria-label="Loading"
    />
  );
}

/** Centred dialog with a dimmed backdrop; Escape and a backdrop click close it. */
export function Modal({
  title,
  onClose,
  children,
  footer,
  width = 520,
}: {
  title: ReactNode;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  width?: number;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className="w-full max-h-[90vh] flex flex-col bg-surface-2 border border-surface-4 rounded-xl shadow-2xl"
        style={{ maxWidth: width }}
        role="dialog"
      >
        <div className="flex items-center gap-3 px-5 h-14 border-b border-surface-4 shrink-0">
          <h2 className="text-base font-semibold text-text-primary truncate">{title}</h2>
          <IconButton icon={X} label="Close" onClick={onClose} className="ml-auto" />
        </div>
        <div className="p-5 overflow-y-auto">{children}</div>
        {footer && (
          <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-surface-4 shrink-0">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}

export interface TabDef<K extends string> {
  key: K;
  label: string;
  icon?: LucideIcon;
  count?: number;
}

export function Tabs<K extends string>({
  tabs,
  active,
  onChange,
  className = "",
}: {
  tabs: TabDef<K>[];
  active: K;
  onChange: (k: K) => void;
  className?: string;
}) {
  return (
    <div className={`flex flex-wrap gap-1 ${className}`} role="tablist">
      {tabs.map(({ key, label, icon: Icon, count }) => {
        const on = key === active;
        return (
          <button
            key={key}
            role="tab"
            aria-selected={on}
            onClick={() => onChange(key)}
            className={`inline-flex items-center gap-1.5 h-8 px-2.5 rounded-md text-xs font-medium whitespace-nowrap transition-colors cursor-pointer
              ${on ? "bg-surface-3 text-text-primary" : "text-text-secondary hover:text-text-primary hover:bg-surface-3/60"}`}
          >
            {Icon && <Icon size={14} />}
            {label}
            {count !== undefined && count > 0 && (
              <span className={`text-[10px] tabular-nums ${on ? "text-accent-purple" : "text-text-muted"}`}>{count}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  children,
  action,
}: {
  icon: LucideIcon;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center text-center gap-2 py-10 px-6">
      <div className="w-11 h-11 rounded-full bg-surface-3 flex items-center justify-center text-text-muted mb-1">
        <Icon size={20} />
      </div>
      <div className="text-sm font-medium text-text-primary">{title}</div>
      {children && <div className="text-xs text-text-secondary max-w-xs leading-relaxed">{children}</div>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

/** Page header used by the full-page views (library, tools). */
export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end gap-x-4 gap-y-3 mb-6">
      <div className="min-w-0">
        <h1 className="text-2xl font-semibold tracking-tight text-text-primary">{title}</h1>
        {subtitle && <p className="text-sm text-text-secondary mt-1">{subtitle}</p>}
      </div>
      {actions && <div className="ml-auto flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}
