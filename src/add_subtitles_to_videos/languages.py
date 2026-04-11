from __future__ import annotations

from dataclasses import dataclass


AUTO_LANGUAGE_CODE = "auto"


@dataclass(frozen=True, slots=True)
class LanguageOption:
    code: str
    label: str


LAUNCH_LANGUAGES: tuple[LanguageOption, ...] = (
    LanguageOption("en", "English"),
    LanguageOption("el", "Greek"),
    LanguageOption("tr", "Turkish"),
    LanguageOption("de", "German"),
    LanguageOption("fr", "French"),
    LanguageOption("it", "Italian"),
    LanguageOption("es", "Spanish"),
    LanguageOption("pt", "Portuguese"),
    LanguageOption("nl", "Dutch"),
    LanguageOption("ro", "Romanian"),
    LanguageOption("pl", "Polish"),
    LanguageOption("cs", "Czech"),
)


AUTO_LANGUAGE = LanguageOption(AUTO_LANGUAGE_CODE, "Auto detect")


LANGUAGE_BY_CODE: dict[str, LanguageOption] = {
    AUTO_LANGUAGE.code: AUTO_LANGUAGE,
    **{language.code: language for language in LAUNCH_LANGUAGES},
}


def source_language_options() -> tuple[LanguageOption, ...]:
    return (AUTO_LANGUAGE, *LAUNCH_LANGUAGES)


def target_language_options() -> tuple[LanguageOption, ...]:
    return LAUNCH_LANGUAGES


def language_label(code: str | None) -> str:
    if code is None:
        return "Unknown"
    option = LANGUAGE_BY_CODE.get(code)
    if option is not None:
        return option.label
    return code
