import React, { useState, useEffect } from "react";
import Spinner from "../common/Spinner";
import GenericSample from "./GenericSample";
import { BASE_URL } from "../../services/mbx";
import "./SampleContainer.css";

function SampleContainer(props) {
  const explorer = props.explorer;
  const sampleId = props.sampleId;

  const [sample, setSample] = useState(null);

  useEffect(() => {
    let isCancelled = false;
    (async () => {
      const params = new URLSearchParams({
        workspace_id: explorer.workspace.id,
        explorer_id: explorer.id,
        sample_id: sampleId,
      });
      const url = `${BASE_URL}/sample?${params.toString()}`;
      const r = await fetch(url);
      const sample = await r.json();
      if (!isCancelled) {
        sample.id = sampleId;
        setSample(sample);
      }
    })();
    return () => (isCancelled = true);
  }, [explorer, sampleId]);

  const content =
    sample == null ? (
      <Spinner />
    ) : (
      <GenericSample sample={sample} explorer={explorer} />
    );
  return <div className="sample-container">{content}</div>;
}

export default SampleContainer;
