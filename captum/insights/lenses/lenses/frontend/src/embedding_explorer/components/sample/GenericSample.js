import React from "react";
import "./GenericSample.css";
import { ImageLoader } from "../common/ImageLoader";
import { BASE_URL } from "../../services/mbx";

const AttrType = Object.freeze({
  IMAGE: "image",
  AUDIO: "audio",
});

function GenericSample(props) {
  const { sample, explorer } = props;
  const rows = [];
  let i = 0;
  const payload = sample.payload;
  for (const attrName in payload) {
    let value = payload[attrName];
    let attrContent;
    if (value?.type != null) {
      const params = new URLSearchParams({
        workspace_id: explorer.workspace.id,
        explorer_id: explorer.id,
        sample_id: sample.id,
        attr_name: attrName,
      });
      const url = `${BASE_URL}/generic_sample_attr?${params.toString()}`;
      switch (value.type) {
        case AttrType.IMAGE:
          attrContent = <ImageLoader alt={attrName} src={url} />;
          break;
        case AttrType.AUDIO:
          attrContent = (
            <div>
              <audio controls>
                <source src={url} />
              </audio>
            </div>
          );
          break;
        default:
          attrContent = value.toString();
      }
    } else {
      attrContent = value.toString();
    }
    const row = (
      <div key={i} className="generic-sample-row">
        <div className="generic-sample-attr-name">{attrName}</div>
        <div className="generic-sample-attr-content">{attrContent}</div>
      </div>
    );
    rows.push(row);
    i++;
  }
  return <div className="generic-sample">{rows}</div>;
}

export default GenericSample;
