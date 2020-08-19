export class RGBColor {
  constructor(r, g, b) {
    this.r = r;
    this.g = g;
    this.b = b;
  }

  lerp(b, t) {
    return new RGBColor(
      this.r * (1 - t) + b.r * t,
      this.g * (1 - t) + b.g * t,
      this.b * (1 - t) + b.b * t,
    );
  }

  toString() {
    return `rgb(${this.r}, ${this.g}, ${this.b})`;
  }
}

RGBColor.white = new RGBColor(255, 255, 255);
RGBColor.darkGray = new RGBColor(217, 217, 217);
RGBColor.lightGray = new RGBColor(242, 242, 242);
RGBColor.darkOrange = new RGBColor(235, 98, 16);
RGBColor.lightOrange = new RGBColor(251, 222, 205);
