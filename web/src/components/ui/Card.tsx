// Card primitive. Composes the `.card`, `.card__header`,
// `.card__title`, `.card__body` classes from app.css so the same
// visual chrome is used for the control panel, time-series cards,
// and the scene overlay.

import type { ReactNode } from "react";

interface CardProps {
  children: ReactNode;
  className?: string;
  elevated?: boolean;
}

export function Card({ children, className = "", elevated }: CardProps) {
  return (
    <div
      className={`card ${elevated ? "card--elevated" : ""} ${className}`.trim()}
    >
      {children}
    </div>
  );
}

interface CardHeaderProps {
  title: ReactNode;
  trailing?: ReactNode;
}

export function CardHeader({ title, trailing }: CardHeaderProps) {
  return (
    <div className="card__header">
      <span className="card__title">{title}</span>
      {trailing ? <span>{trailing}</span> : null}
    </div>
  );
}

interface CardBodyProps {
  children: ReactNode;
  className?: string;
}

export function CardBody({ children, className = "" }: CardBodyProps) {
  return <div className={`card__body ${className}`.trim()}>{children}</div>;
}
