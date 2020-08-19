import { Lens } from "../models/lens";

export async function getActiveLens(): Promise<Lens> {
  const response = await fetch("/lens");
  if (!response.ok) {
    throw Error("Failed to get active lens");
  }

  return response.json();
}
