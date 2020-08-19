import React, { useState, useEffect } from "react";
import Workspace from "../models/Workspace";
import Spinner from "./common/Spinner";
import ExplorerContainer from "./ExplorerContainer";
import { BASE_URL } from "../services/mbx";
import "./WorkspaceContainer.css";

function WorkspaceContainer(props) {
  if (props.id == null) throw new Error("workspace id required");

  const [workspace, setWorkspace] = useState(null);
  const [comparisonMode, setComparisonMode] = useState(false);
  const [selectionContext, setSelectionContext] = useState({});

  useEffect(() => {
    (async () => {
      const params = new URLSearchParams({
        id: props.id,
      });
      const url = `${BASE_URL}/workspace?${params.toString()}`;
      const r = await fetch(url);
      const data = await r.json();
      const workspace = Workspace.fromData(data);
      if (workspace.id == null) throw new Error("workspace id required");
      setWorkspace(workspace);
    })();
  }, [props.id]);

  const onComponentClick = (component, e) => {
    // TODO clean up temporary controls (shift + click)
    // to select / deselect component at the workspace level,
    // we have not decided how to display correlations in the new design yet
    if (e.shiftKey) {
      if (component.equal(selectionContext.selectedComponent)) {
        setSelectionContext({});
      } else {
        setSelectionContext({
          selectedComponent: component,
        });
      }
    }
  };

  const onCompare = (idx) => setComparisonMode(true);
  const onClose = (idx) => setComparisonMode(false);

  let content;
  if (workspace == null) {
    content = <Spinner />;
  } else {
    let explorers;
    // TODO generalize to more than 2 explorers
    if (comparisonMode) {
      explorers = [workspace.explorers[0], workspace.explorers[1]];
    } else {
      explorers =
        workspace.explorers.length === 0 ? [] : [workspace.explorers[0]];
    }
    content = (
      <div className="workspace-explorers">
        {explorers.map((explorer, i) => (
          <ExplorerContainer
            key={i}
            idx={i}
            id={explorer.id}
            workspace={workspace}
            componentMode={props.componentMode}
            sampleWindowSize={props.sampleWindowSize}
            minSampleWindowSize={props.minSampleWindowSize}
            maxSampleWindowSize={props.maxSampleWindowSize}
            comparisonMode={comparisonMode}
            allowComparison={workspace.explorers.length > 1}
            onCompare={onCompare}
            onClose={onClose}
            onComponentClick={onComponentClick}
            workspaceSelectionContext={selectionContext}
          />
        ))}
      </div>
    );
  }

  return <div className="workspace-container">{content}</div>;
}

export default WorkspaceContainer;
