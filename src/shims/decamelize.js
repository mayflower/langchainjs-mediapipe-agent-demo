export default function decamelize(input, separator = "_") {
  if (typeof input !== "string") {
    throw new TypeError("Expected a string");
  }

  return input
    .replace(/([a-z\d])([A-Z])/g, `$1${separator}$2`)
    .replace(/([A-Z]+)([A-Z][a-z\d]+)/g, `$1${separator}$2`)
    .toLowerCase();
}
