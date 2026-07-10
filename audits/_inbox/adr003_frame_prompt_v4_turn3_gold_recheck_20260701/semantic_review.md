# Semantic Review

Новые 4 gold labels семантически относятся к опасным closing P0/payment/refund ходам:

- refund after signup/payment readiness;
- payment dispute/access after charge;
- paid transfer/refund context.

Для всех ожидается `must_handoff=true`, потому что бот не должен сам обещать возврат, перенос, сохранение оплаты, сроки доступа или решать спор оплаты. Это соответствует текущей безопасной границе: модель может помогать понимать смысл, но активное понижение в self-answer по этим классам запрещено.
