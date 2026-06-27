# Риски

- Клиентский риск: closing-fix теперь включается профилем, значит влияет на живой pilot-profile direct-path. Риск закрыт тестами на profile-on и P0-silent, но нужен финальный регрейд Claude #1 на реальных/симулированных closing-кейсах.
- P0-гигиена: default OFF, в профиль не добавлена. Если включать позже, основной риск — слишком широкое переписывание полезного manager-only текста. Покрыты refund/payment_dispute/presale cases, но перед включением нужен отдельный ON/OFF регрейд.
- Данные/записи: AMO/Tallanto/CRM/client write не запускались. Код меняет только генерацию черновика в памяти процесса.
- Откат: убрать `TONE_CLOSE_DETECT_ENV` из `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS` для отключения closing-fix; P0-гигиена уже выключена по умолчанию.
