import React from "react";
import { RGBColor } from "../utils/color";
import "./ComponentBars.css";

function ComponentBar(props) {
  const maxLength = 30;
  const thickness = 20;
  const component = props.component;
  const value = component.value;
  const length = Math.max(1, value * maxLength);

  let color = RGBColor.darkOrange;
  const updateColor = props.updateColor ?? true;

  const selectedExplorerComponent =
    props.explorerSelectionContext?.selectedComponent;

  // TODO temporarily ignoring workspace selection,
  // but we should consider using selectedWorkspaceComponent,
  // if available, to show correlations
  // const selectedWorkspaceComponent =
  //   props.workspaceSelectionContext == null
  //     ? null
  //     : props.workspaceSelectionContext.selectedComponent;
  // const selectedComponent =
  //   selectedWorkspaceComponent ?? selectedExplorerComponent;
  const selectedComponent = selectedExplorerComponent;

  if (updateColor) {
    if (selectedComponent == null) {
      color = RGBColor.darkOrange;
    } else {
      if (selectedComponent.module.explorer === component.module.explorer) {
        if (selectedComponent.module !== component.module) {
          const correlations = component.module.explorer.getComponentCorrelations(
            props.componentMode,
          );
          const correlation = correlations.get(component, selectedComponent);
          color = RGBColor.lightOrange.lerp(
            RGBColor.darkOrange,
            Math.abs(correlation),
          );
        } else {
          color =
            selectedComponent.id === component.id
              ? RGBColor.darkOrange
              : RGBColor.lightOrange;
        }
      } else {
        const correlations = component.module.explorer.workspace.getComponentCorrelations(
          props.componentMode,
        );
        const correlation = correlations.get(component, selectedComponent);
        color = RGBColor.lightOrange.lerp(
          RGBColor.darkOrange,
          Math.abs(correlation),
        );
      }
    }
  }

  let barStyle = {
    backgroundColor: color.toString(),
  };

  if (selectedComponent != null && selectedComponent.equals(component)) {
    barStyle.border = "1px solid black";
  }

  if (props.direction === "row") {
    barStyle.width = `${thickness}px`;
    barStyle.height = `${length}px`;
    barStyle.marginTop = "auto";
  } else {
    barStyle.height = `${thickness}px`;
    barStyle.width = `${length}px`;
  }

  return (
    <div
      className="component-bar"
      style={barStyle}
      onClick={props.onClick}
    ></div>
  );
}

function ComponentBars(props) {
  const direction = props.direction ?? "column";
  const className = `component-bars-${direction}`;

  return (
    <div className={className}>
      {props.components.map((component, i) => {
        const onClick = (e) => {
          e.stopPropagation();
          props.onComponentClick && props.onComponentClick(component);
        };
        return (
          <ComponentBar
            key={i}
            component={component}
            direction={props.direction}
            onClick={onClick}
            updateColor={props.updateColor}
            componentMode={props.componentMode}
            explorerSelectionContext={props.explorerSelectionContext}
            workspaceSelectionContext={props.workspaceSelectionContext}
          />
        );
      })}
    </div>
  );
}

export default ComponentBars;
