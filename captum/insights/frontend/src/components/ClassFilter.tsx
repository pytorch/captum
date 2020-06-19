import React from "react";
import ReactTags from "react-tag-autocomplete";

function ClassFilter(props) {
  return (
    <ReactTags
      tags={props.classes}
      autofocus={false}
      suggestions={props.suggestedClasses}
      handleDelete={props.handleClassDelete}
      handleAddition={props.handleClassAdd}
      minQueryLength={0}
      placeholder="add new class..."
    />
  );
}

export default ClassFilter;
