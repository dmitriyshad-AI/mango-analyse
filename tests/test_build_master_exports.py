from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest
from datetime import date
from pathlib import Path


MODULE_PATH = Path('/Users/dmitrijfabarisov/Projects/Mango analyse/scripts/build_messages28_master_exports.py')
spec = importlib.util.spec_from_file_location('build_messages28_master_exports', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class BuildMasterExportsTest(unittest.TestCase):
    def test_build_contact_history_does_not_truncate_latest_five_lines(self) -> None:
        long_summary = 'Очень подробное содержательное резюме разговора ' + ('с деталями ' * 40)
        long_step = 'Нужно отправить материалы и согласовать следующий созвон ' + ('после уточнения условий ' * 12)
        calls = []
        for idx in range(6):
            calls.append(
                {
                    'Дата и время звонка': f'2026-04-0{idx + 1} 10:00:00',
                    'Менеджер': 'Тестовый Менеджер',
                    'Тип звонка': 'sales_call',
                    'Статус Analyze': 'done',
                    'Краткое резюме разговора': f'{long_summary} #{idx}',
                    'Следующий шаг': f'{long_step} #{idx}',
                    'Возражения': '',
                    'Имя исходного файла': f'call_{idx}.mp3',
                }
            )

        short_history, chronology = module._build_contact_history(calls)

        self.assertIn('Контакт в истории с 01.04.2026 по 06.04.2026', short_history)
        self.assertIn('Проанализировано звонков: 6', short_history)
        self.assertIn('Последний содержательный контекст:', short_history)
        self.assertIn('Текущий согласованный следующий шаг:', short_history)
        self.assertIn('#5', chronology)
        self.assertIn('#1', chronology)
        self.assertNotIn('…', chronology)
        self.assertNotEqual(short_history, chronology)
        self.assertNotIn('01.04.2026 — Тестовый Менеджер (sales_call):', short_history)

    def test_adjust_contact_operational_fields_accounts_for_stale_last_contact(self) -> None:
        follow_up, priority, probability = module._adjust_contact_operational_fields(
            analysis_date=date(2026, 4, 13),
            last_contact_raw='2026-03-20 14:00:00',
            follow_up_raw='2026-03-25',
            priority_raw='hot',
            probability_raw='80',
        )

        self.assertEqual(follow_up, '2026-04-13')
        self.assertEqual(priority, 'warm')
        self.assertEqual(probability, '65')

    def test_adjust_contact_operational_fields_sets_follow_up_for_recent_warm_contact_without_date(self) -> None:
        follow_up, priority, probability = module._adjust_contact_operational_fields(
            analysis_date=date(2026, 4, 13),
            last_contact_raw='2026-04-12 09:00:00',
            follow_up_raw='',
            priority_raw='warm',
            probability_raw='60',
        )

        self.assertEqual(follow_up, '2026-04-13')
        self.assertEqual(priority, 'warm')
        self.assertEqual(probability, '60')

    def test_build_contact_summary_aggregates_products_managers_and_objections(self) -> None:
        calls = [
            {
                'Дата и время звонка': '2026-04-12 09:00:00',
                'Менеджер': 'Анна',
                'Тип звонка': 'sales_call',
                'Статус Analyze': 'done',
                'Краткое резюме разговора': 'Клиент хочет продолжить обучение в следующем году.',
                'Следующий шаг': 'Отправить материалы',
                'Возражения': 'цена | расписание',
                'Продукты интереса': 'годовые курсы | информатика',
                'Рекомендуемый продукт': 'годовые курсы',
                'Имя исходного файла': 'a.mp3',
            },
            {
                'Дата и время звонка': '2026-04-10 10:00:00',
                'Менеджер': 'Олег',
                'Тип звонка': 'sales_call',
                'Статус Analyze': 'done',
                'Краткое резюме разговора': 'Клиент просит вернуться после уточнения расписания.',
                'Следующий шаг': 'Перезвонить в пятницу',
                'Возражения': 'расписание',
                'Продукты интереса': 'математика',
                'Рекомендуемый продукт': 'годовые курсы',
                'Имя исходного файла': 'b.mp3',
            },
        ]

        short_history, chronology = module._build_contact_history(calls)

        self.assertIn('Менеджеры: Анна, Олег.', short_history)
        self.assertIn('Ключевой интерес: годовые курсы, информатика, математика.', short_history)
        self.assertIn('Повторяющиеся ограничения/возражения: цена, расписание.', short_history)
        self.assertIn('Текущий согласованный следующий шаг: Отправить материалы.', short_history)
        self.assertIn('12.04.2026 — Анна (sales_call):', chronology)

    def test_repair_analysis_for_export_restores_truncated_history_summary(self) -> None:
        call = SimpleNamespace(
            started_at=module._parse_dt('2026-04-12 10:00:00'),
            manager_name='Анна',
            phone='+79990000001',
            direction='outbound',
            source_file='a.mp3',
            source_filename='a.mp3',
            transcript_variants_json=None,
        )
        long_tail = ' '.join(f'деталь{i}' for i in range(180))
        analysis = {
            'history_summary': ('Клиент обсуждал подробности программы. ' + long_tail)[:1097].rstrip() + '...',
            'summary': 'Менеджер подробно разъяснил программу и договорился отправить материалы.',
            'structured_fields': {
                'people': {},
                'contacts': {},
                'student': {},
                'interests': {'products': ['годовые курсы'], 'subjects': ['математика'], 'format': [], 'exam_targets': []},
                'commercial': {},
                'objections': [],
                'next_step': {'action': 'Отправить материалы', 'due': None},
                'lead_priority': 'warm',
            },
            'follow_up_score': 60,
            'follow_up_reason': 'Есть согласованный следующий шаг.',
        }
        transcript_text = 'MANAGER:\\nПодробно рассказал о программе.\\nCLIENT:\\nХорошо, пришлите материалы и полное описание.'

        repaired = module._repair_analysis_for_export(call, transcript_text, analysis)

        self.assertIn('отправить материалы', repaired.get('history_summary', '').lower())
        self.assertNotIn('...', str(repaired.get('history_summary', '')))


if __name__ == '__main__':
    unittest.main()
