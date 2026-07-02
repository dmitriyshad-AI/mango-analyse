# Risk Review

## Scope

Report-only review. Runtime-код не менялся.

## Safety

- Live Telegram bot не тронут.
- Wappi не тронут.
- AMO/CRM/Tallanto не тронуты.
- Профиль и флаги не менялись.
- P0-floor/preblock не менялись.

## Main Risk Found

Нельзя дальше усиливать prompt так, чтобы frame называл safe/answer_self без
проверенного факта. Это даст ложную автономность и риск выдумки.

## Guardrail

Любая будущая активная автономность должна проходить через проверяемый
fresh client-safe exact proof, а не через доверие к тексту модели.
