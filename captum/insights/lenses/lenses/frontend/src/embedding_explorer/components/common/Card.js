import React from "react";
import "./Card.css";

function Card(props) {
  let style = {};
  if (props.maxHeight != null) style.maxHeight = props.maxHeight;
  // TODO remove mbx prefix, merge with CNN inspector card
  return (
    <div className="mbx-card" style={style}>
      <div className="mbx-card-title">{props.title}</div>
      <div className="mbx-card-body">{props.children}</div>
    </div>
  );
}

export default Card;
