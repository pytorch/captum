import React from "react";
import ReactTags from "react-tag-autocomplete";
import { TagClass } from "../models/filter";

interface ClassFilterProps {
  suggestedClasses: TagClass[];
  classes: TagClass[];
  handleClassDelete: (classId: number) => void;
  handleClassAdd: (newClass: TagClass) => void;
}

function ClassFilter(props: ClassFilterProps) {
  const handleAddition = (newTag: { id: number | string; name: string }) => {
    /**
     * Need this type check as we expect tagId to be number while the `react-tag-autocomplete` has
     * id as number | string.
     */
    if (typeof newTag.id === "string") {
      throw Error("Invalid tag id received from ReactTags");
    } else {
      props.handleClassAdd({ id: newTag.id, name: newTag.name });
    }
  };

  return (
    <ReactTags
      tags={props.classes}
      autofocus={false}
      suggestions={props.suggestedClasses}
      handleDelete={props.handleClassDelete}
      handleAddition={handleAddition}
      minQueryLength={0}
      placeholder="add new class..."
    />
  );
}

export default ClassFilter;
