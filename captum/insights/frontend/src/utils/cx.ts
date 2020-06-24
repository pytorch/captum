// helper method to convert an array or object into a valid classname
function cx(obj: any) {
  if (Array.isArray(obj)) {
    return obj.join(" ");
  }
  return Object.keys(obj)
    .filter((k) => !!obj[k])
    .join(" ");
}

export default cx;
