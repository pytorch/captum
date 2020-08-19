import React from "react";
import { RGBColor } from "../utils/color";
import "./CorrelatedComponents.css";

function Link(props) {
  const thickness = props.thickness ?? 1.0;
  const opacity = props.opacity ?? 1.0;
  const color = props.color ?? RGBColor.darkOrange;
  const { x, y } = props.a;
  const [w, h] = [props.b.x - x, props.b.y - y];
  const d = `M ${x} ${y} Q ${x + w * 0.25} ${y}, ${x + w * 0.5} ${
    y + h * 0.5
  } T ${x + w} ${y + h}`;
  return (
    <path
      className="corr-link"
      d={d}
      stroke={color}
      opacity={opacity}
      strokeWidth={`${thickness}px`}
      fill="none"
    />
  );
}

function Component(props) {
  const barWidth = props.barWidth ?? 8;
  const padding = props.padding ?? 16;
  const strokeWidth = props.strokeWidth ?? 1;
  const maxBarHeight = (props.maxBarHeight ?? 24) - strokeWidth;
  const component = props.component;
  const value = component.value;
  const barHeight = maxBarHeight * value;
  const labelMarginLeft = props.labelMarginLeft ?? 12;
  const bgBarColor = props.bgBarColor ?? RGBColor.lightGray;
  const fgBarColor = props.fgBarColor ?? RGBColor.darkOrange;
  const backgroundOpacity = props.backgroundOpacity ?? 0.6;
  const selectedBackgroundOpacity = props.backgroundOpacity ?? 0.6;

  let background = null;
  const backgroundStrokeWidth = 0;
  const selectedComponent = props.selectedComponent;
  if (selectedComponent != null) {
    let correlationColor = null;
    if (selectedComponent.equals(component)) {
      correlationColor = RGBColor.darkOrange;
    } else {
      const correlation = component.getCorrelation(props.selectedComponent);
      if (correlation != null) {
        correlationColor = RGBColor.lightGray.lerp(
          RGBColor.darkOrange,
          Math.abs(correlation),
        );
      }
    }
    if (correlationColor != null) {
      const backgroundWidth = barWidth + padding;
      const backgroundHeight = maxBarHeight + padding;
      background = (
        <rect
          width={backgroundWidth}
          height={backgroundHeight}
          fill={correlationColor}
          fillOpacity={
            component.equals(selectedComponent)
              ? selectedBackgroundOpacity
              : backgroundOpacity
          }
          x={props.x - backgroundWidth * 0.5}
          y={props.y - backgroundHeight * 0.5}
          stroke={
            component.equals(selectedComponent) ? RGBColor.darkOrange : null
          }
          strokeWidth={backgroundStrokeWidth}
        />
      );
    }
  }

  const bgBar = (
    <rect
      x={props.x - barWidth * 0.5}
      y={props.y - maxBarHeight * 0.5}
      fill={bgBarColor}
      height={maxBarHeight}
      width={barWidth}
      stroke="black"
      strokeWidth={strokeWidth}
    />
  );
  const fgBar = (
    <rect
      x={props.x - (barWidth - strokeWidth) * 0.5}
      y={props.y + maxBarHeight * 0.5 - barHeight}
      fill={fgBarColor}
      height={barHeight}
      width={barWidth - strokeWidth}
    />
  );

  const label = `${component.type.componentPrefix} ${component.id + 1}`;
  const onClick = () => props.onClick && props.onClick();

  return (
    <g onClick={onClick} cursor="pointer">
      <g>
        {background}
        {bgBar}
        {fgBar}
      </g>
      <text
        alignmentBaseline="middle"
        x={props.x + barWidth * 0.5 + labelMarginLeft}
        y={props.y}
        fontWeight={component.equals(selectedComponent) ? "bold" : null}
        fill={
          component.equals(selectedComponent) ? RGBColor.darkOrange : "black"
        }
      >
        {label}
      </text>
    </g>
  );
}

function CorrelatedComponents(props) {
  const explorerSelectionContext = props.explorerSelectionContext ?? {};
  const modules = props.modules ?? [];
  const padding = props.padding ?? 48;
  const maxCorrelationThickness = props.maxCorrelationThickness ?? 24;
  const dx = props.dx ?? 120; // horizontal space between modules
  const dy = props.dy ?? 70; // vertical space between components
  const dy0 = props.dy0 ?? 40; // vertical space between module label and first component
  const activeCorrelationOpacity = props.activeCorrelationOpacity ?? 0.3;
  const inactiveCorrelationOpacity = props.inactiveCorrelationOpacity ?? 0.05;

  const selectedComponent = explorerSelectionContext.selectedComponent;

  const nodes = []; // svg elements used to show component data
  const links = []; // svg elements used to show correlation between components
  const moduleLabels = [];

  // to keep track of horizontal and vertical space used
  // and set the width and height of the svg container
  let [maxX, maxY] = [0, 0];

  modules.forEach((module, i) => {
    const x = i * dx;
    const y = 0;
    const label = (
      <text key={i} x={x} y={y} textAnchor="middle">
        {module.name}
      </text>
    );

    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;

    moduleLabels.push(label);

    const nextModule = i < modules.length - 1 ? modules[i + 1] : null;
    const components = module.getComponents(props.componentMode);

    components.forEach((component, j) => {
      const y = dy0 + j * dy;

      const onComponentClick = () =>
        props.onComponentClick && props.onComponentClick(component);

      const node = (
        <Component
          key={nodes.length}
          barWidth={12}
          barMaxHeight={23}
          x={x}
          y={y}
          component={component}
          onClick={onComponentClick}
          selectedComponent={selectedComponent}
          bgBarColor={RGBColor.darkGray}
          strokeWidth={0}
        />
      );

      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;

      nodes.push(node);

      if (nextModule == null) return;

      const x2 = (i + 1) * dx;
      const components2 = nextModule.getComponents(props.componentMode);

      components2.forEach((component2, j2) => {
        const correlation = component.getCorrelation(component2);
        if (correlation != null) {
          const thickness = maxCorrelationThickness * Math.abs(correlation);
          const y2 = dy0 + j2 * dy;
          let color, opacity;

          if (
            selectedComponent == null ||
            component.equals(selectedComponent) ||
            component2.equals(selectedComponent)
          ) {
            color = RGBColor.darkOrange;
            opacity = activeCorrelationOpacity;
          } else {
            color = new RGBColor(30, 30, 30);
            opacity = inactiveCorrelationOpacity;
          }
          const link = (
            <Link
              key={links.length}
              color={color}
              opacity={opacity}
              a={{ x: x, y: y }}
              b={{ x: x2, y: y2 }}
              thickness={thickness}
            />
          );
          links.push(link);
        }
      });
    });
  });

  return (
    <div className="correlated-components">
      <svg
        className="correlated-components-svg"
        style={{
          width: `${maxX}px`,
          height: `${maxY}px`,
          padding: `${padding}px`,
        }}
      >
        <g>{links}</g>
        <g>{nodes}</g>
        <g>{moduleLabels}</g>
      </svg>
    </div>
  );
}

export default CorrelatedComponents;
