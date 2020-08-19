import React from "react";
import "./Spinner.css";

// TODO merge with CNN explorer spinner

function Spinner(props) {
  const size = props.size ?? 46;
  const borderWidth = props.borderWidth ?? 5;
  const color = props.color ?? "#ee4c2c";
  const style = {
    width: `${size}px`,
    height: `${size}px`,
    borderWidth: `${borderWidth}px`,
    borderColor: `${color} transparent ${color} transparent`,
  };

  return (
    <div className="spinner-wrapper">
      <div className="spinner-icon" style={style} />
    </div>
  );
}

export default Spinner;
