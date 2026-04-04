const UPPERCASE = /[\p{Lu}]/u;
const LOWERCASE = /[\p{Ll}]/u;
const LEADING_CAPITAL = /^[\p{Lu}](?![\p{Lu}])/gu;
const IDENTIFIER = /([\p{Alpha}\p{N}_]|$)/u;
const SEPARATORS = /[_.\- ]+/;

const LEADING_SEPARATORS = new RegExp(`^${SEPARATORS.source}`);
const SEPARATORS_AND_IDENTIFIER = new RegExp(
  `${SEPARATORS.source}${IDENTIFIER.source}`,
  "gu"
);
const NUMBERS_AND_IDENTIFIER = new RegExp(`\\d+${IDENTIFIER.source}`, "gu");

function preserveCamelCase(input, toLowerCase, toUpperCase) {
  let isLastCharLower = false;
  let isLastCharUpper = false;
  let isLastLastCharUpper = false;

  for (let index = 0; index < input.length; index += 1) {
    const character = input[index];

    if (isLastCharLower && UPPERCASE.test(character)) {
      input = `${input.slice(0, index)}-${input.slice(index)}`;
      isLastCharLower = false;
      isLastLastCharUpper = isLastCharUpper;
      isLastCharUpper = true;
      index += 1;
    } else if (
      isLastCharUpper &&
      isLastLastCharUpper &&
      LOWERCASE.test(character)
    ) {
      input = `${input.slice(0, index - 1)}-${input.slice(index - 1)}`;
      isLastLastCharUpper = isLastCharUpper;
      isLastCharUpper = false;
      isLastCharLower = true;
    } else {
      isLastCharLower =
        toLowerCase(character) === character &&
        toUpperCase(character) !== character;
      isLastLastCharUpper = isLastCharUpper;
      isLastCharUpper =
        toUpperCase(character) === character &&
        toLowerCase(character) !== character;
    }
  }

  return input;
}

function preserveConsecutiveUppercase(input, toLowerCase) {
  LEADING_CAPITAL.lastIndex = 0;
  return input.replace(LEADING_CAPITAL, (value) => toLowerCase(value));
}

function postProcess(input, toUpperCase) {
  SEPARATORS_AND_IDENTIFIER.lastIndex = 0;
  NUMBERS_AND_IDENTIFIER.lastIndex = 0;

  return input
    .replace(SEPARATORS_AND_IDENTIFIER, (_, identifier) =>
      toUpperCase(identifier)
    )
    .replace(NUMBERS_AND_IDENTIFIER, (value) => toUpperCase(value));
}

export default function camelcase(input, options = {}) {
  if (!(typeof input === "string" || Array.isArray(input))) {
    throw new TypeError("Expected the input to be `string | string[]`");
  }

  const normalizedOptions = {
    pascalCase: false,
    preserveConsecutiveUppercase: false,
    ...options,
  };

  if (Array.isArray(input)) {
    input = input
      .map((value) => value.trim())
      .filter((value) => value.length)
      .join("-");
  } else {
    input = input.trim();
  }

  if (input.length === 0) {
    return "";
  }

  const toLowerCase =
    normalizedOptions.locale === false
      ? (value) => value.toLowerCase()
      : (value) => value.toLocaleLowerCase(normalizedOptions.locale);
  const toUpperCase =
    normalizedOptions.locale === false
      ? (value) => value.toUpperCase()
      : (value) => value.toLocaleUpperCase(normalizedOptions.locale);

  if (input.length === 1) {
    return normalizedOptions.pascalCase
      ? toUpperCase(input)
      : toLowerCase(input);
  }

  const hasUpperCase = input !== toLowerCase(input);
  if (hasUpperCase) {
    input = preserveCamelCase(input, toLowerCase, toUpperCase);
  }

  input = input.replace(LEADING_SEPARATORS, "");
  input = normalizedOptions.preserveConsecutiveUppercase
    ? preserveConsecutiveUppercase(input, toLowerCase)
    : toLowerCase(input);

  if (normalizedOptions.pascalCase) {
    input = `${toUpperCase(input.charAt(0))}${input.slice(1)}`;
  }

  return postProcess(input, toUpperCase);
}
