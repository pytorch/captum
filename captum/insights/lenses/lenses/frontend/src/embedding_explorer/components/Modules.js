import React from "react";
import ComponentBars from "./ComponentBars";
import { RGBColor } from "../utils/color";
import "./Modules.css";

function Modules(props) {
  const modules = props.modules;

  const componentBars = [];
  const names = [];

  const onComponentClick = (component) => {
    props.onComponentClick && props.onComponentClick(component);
  };

  modules.forEach((module, i) => {
    names.push(module.name);
    const components = module.getComponents(props.componentMode);
    componentBars.push(
      <ComponentBars
        components={components}
        direction={props.componentDirection}
        onComponentClick={onComponentClick}
        updateColor={props.updateColor}
        componentMode={props.componentMode}
        explorerSelectionContext={props.explorerSelectionContext}
        workspaceSelectionContext={props.workspaceSelectionContext}
      />,
    );
  });

  const rows = [];
  let n = modules.length;
  const padding = props.padding == null ? "10px" : props.padding;
  const direction = props.direction == null ? "row" : props.direction;

  const selectionContext = props.selectionContext;

  if (direction === "row") {
    let cells = [];
    for (let i = 0; i < n; i++) {
      const module = modules[i];

      let nameCellStyle = {
        padding,
        cursor: "pointer",
      };
      const selectedModule = selectionContext.selectedModule;
      if (selectedModule != null) {
        if (module.explorer.id === selectedModule.explorer.id) {
          if (module.id === selectedModule.id) {
            nameCellStyle.backgroundColor = RGBColor.darkOrange.toString();
            nameCellStyle.border = "1px solid black";
          } else {
            nameCellStyle.backgroundColor = RGBColor.white.toString();
          }
        } else {
          const workspace = module.explorer.workspace;
          const correlation = workspace.moduleCorrelations.get(
            module,
            selectedModule,
          );
          // CCA score can return arbitrary negative values and positive values <= 1
          const intensity = Math.max(0, correlation);
          nameCellStyle.backgroundColor = RGBColor.white.lerp(
            RGBColor.darkOrange,
            intensity,
          );
        }
      }
      const onModuleClick = () => {
        props.onModuleClick && props.onModuleClick(module);
      };
      cells.push(
        <div key={i} className="table-cell-display">
          <div style={nameCellStyle} onClick={onModuleClick}>
            {names[i]}
          </div>
        </div>,
      );
    }
    rows.push(
      <div key={0} className="table-row-display">
        {cells}
      </div>,
    );

    cells = [];
    for (let i = 0; i < n; i++) {
      cells.push(
        <div key={i} className="table-cell-display">
          <div style={{ padding }}>{componentBars[i]}</div>
        </div>,
      );
    }
    rows.push(
      <div key={1} className="table-row-display">
        {cells}
      </div>,
    );
  } else {
    for (let i = 0; i < n; i++) {
      const module = modules[i];
      const onModuleClick = () => {
        props.onModuleClick(module);
      };
      let cells = [];
      let cellStyle = {};
      const selectedModuleIds = props.selectedModuleIds;
      const selected = selectedModuleIds.indexOf(module.id) !== -1;
      if (selected) cellStyle.backgroundColor = RGBColor.lightOrange.toString();
      cells.push(
        <div key={0} className="table-cell-display" style={cellStyle}>
          <div style={{ padding: padding }}>{names[i]}</div>
        </div>,
      );
      cells.push(
        <div key={1} className="table-cell-display" style={cellStyle}>
          <div style={{ padding: padding }}>{componentBars[i]}</div>
        </div>,
      );

      rows.push(
        <div key={i} className="table-row-display" onClick={onModuleClick}>
          {cells}
        </div>,
      );
    }
  }

  return <div className="table-display">{rows}</div>;
}

export default Modules;
