export function compareNumList(l1, l2) {
  if (l1.length === 0) return 0;
  if (l1[0] < l2[0]) return -1;
  else if (l1[0] > l2[0]) return 1;
  else return compareNumList(l1.slice(1), l2.slice(1));
}

export function hashNumListPair(l1, l2) {
  const ls = compareNumList(l1, l2) < 0 ? [l1, l2] : [l2, l1];
  const key = ls[0].join("/") + "_" + ls[1].join("/");
  return key;
}
