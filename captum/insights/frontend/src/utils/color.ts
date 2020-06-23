function calcHSLFromScore(percentage: number, zeroDefault = false) {
  const blue_hsl = [220, 100, 80];
  const red_hsl = [10, 100, 67];

  let target_hsl = null;
  if (percentage > 0) {
    target_hsl = blue_hsl;
  } else {
    target_hsl = red_hsl;
  }

  const default_hsl = [0, 40, zeroDefault ? 100 : 90];
  const abs_percent = Math.abs(percentage * 0.01);
  if (abs_percent < 0.02) {
    return `hsl(${default_hsl[0]}, ${default_hsl[1]}%, ${default_hsl[2]}%)`;
  }

  const color = [
    target_hsl[0],
    (target_hsl[1] - default_hsl[1]) * abs_percent + default_hsl[1],
    (target_hsl[2] - default_hsl[2]) * abs_percent + default_hsl[2],
  ];
  return `hsl(${color[0]}, ${color[1]}%, ${color[2]}%)`;
}

export { calcHSLFromScore };
