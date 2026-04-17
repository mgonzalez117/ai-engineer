import pandas as pd
import numpy as np
from IPython.display import display
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

class FormatType(Enum):
    NUMBER = "number"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    PERCENTAGE_WITH_SIGN = "percentage_with_sign"
    NOTE = "note"


@dataclass
class RowConfig:
    """Configuration pour une ligne (métrique) du tableau."""
    name: str
    display_name: str
    format_type: FormatType = FormatType.NUMBER
    custom_formatter: Optional[Callable[[float], str]] = None
    format_params: Optional[Dict[str, Any]] = None


class BaseFormatter(ABC):
    @abstractmethod
    def format(self, value: float, **kwargs) -> str:
        pass


class NumberFormatter(BaseFormatter):
    def format(self, value: float, decimals: int = 2, threshold: int = 1000) -> str:
        if pd.isna(value):
            return "—"
        if abs(value) < threshold:
            return f"{value:,.{decimals}f}".replace(',', ' ')
        else:
            return f"{value:,.0f}".replace(',', ' ')


class CurrencyFormatter(BaseFormatter):
    def format(self, value: float, currency: str = "€", decimals: int = 0, threshold: int = 1000) -> str:
        if pd.isna(value):
            return "—"
        if abs(value) < threshold:
            return f"{value:,.2f} {currency}".replace(',', ' ')
        else:
            return f"{value:,.{decimals}f} {currency}".replace(',', ' ')


class PercentageFormatter(BaseFormatter):
    def format(self, value: float, with_sign: bool = False, decimals: int = 1) -> str:
        if pd.isna(value):
            return "—"
        sign = "+" if with_sign and value > 0 else ""
        return f"{sign}{value:.{decimals}f} %"


class NoteFormatter(BaseFormatter):
    def format(self, value: float) -> str:
        if pd.isna(value):
            return "—"
        return f"{value:.1f} / 5"


class FormatterFactory:
    _formatters = {
        FormatType.NUMBER: NumberFormatter(),
        FormatType.CURRENCY: CurrencyFormatter(),
        FormatType.PERCENTAGE: PercentageFormatter(),
        FormatType.PERCENTAGE_WITH_SIGN: PercentageFormatter(),
        FormatType.NOTE: NoteFormatter(),
    }

    @classmethod
    def get_formatter(cls, format_type: FormatType, custom_formatter: Optional[Callable] = None) -> BaseFormatter:
        if custom_formatter:
            class CustomFormatterWrapper(BaseFormatter):
                def format(self, value: float, **kwargs) -> str:
                    return custom_formatter(value)
            return CustomFormatterWrapper()
        return cls._formatters.get(format_type, cls._formatters[FormatType.NUMBER])


class DataProcessor:
    @staticmethod
    def normalize_overtime_hours(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        hs = df['heure_supplementaires']
        text_mapping = {'Oui': 1, 'Non': 0, 'oui': 1, 'non': 0, 'OUI': 1, 'NON': 0}
        df['heure_supplementaires'] = (
            hs.map(text_mapping)
              .fillna(hs.replace({True: 1, False: 0}))
        )
        df['heure_supplementaires'] = pd.to_numeric(df['heure_supplementaires'], errors='coerce')
        df['heure_supplementaires'] = np.where(
            df['heure_supplementaires'].between(0, 1, inclusive='both'),
            df['heure_supplementaires'] * 100,
            df['heure_supplementaires']
        )
        return df

    @staticmethod
    def normalize_salary_increase(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['augementation_salaire_precedente'] = (
            df['augementation_salaire_precedente']
              .astype(str)
              .str.replace('%', '', regex=False)
              .str.replace(',', '.', regex=False)
        )
        df['augementation_salaire_precedente'] = pd.to_numeric(
            df['augementation_salaire_precedente'], errors="coerce"
        )
        return df


class RowManager:
    """Gestionnaire des lignes (métriques) et de leur configuration."""

    def __init__(self, rowsConfig):
        self.rowsConfig = rowsConfig

    def get_row_names(self) -> List[str]:
        return [row.name for row in self.rowsConfig]

    def get_display_names_mapping(self) -> Dict[str, str]:
        return {row.name: row.display_name for row in self.rowsConfig}

    def get_row_config(self, row_name: str) -> Optional[RowConfig]:
        for row in self.rowsConfig:
            if row.name == row_name or row.display_name == row_name:
                return row
        return None

    def get_rows_by_format_type(self, format_type: FormatType) -> List[str]:
        return [row.display_name for row in self.rowsConfig if row.format_type == format_type]


class StyleConfig:
    """Configuration des styles pour le tableau."""
    BOLD_THRESHOLD_HIGH = 1000
    BOLD_THRESHOLD_LOW = 0.3

    NEGATIVE_COLOR = '#d32f2f'
    POSITIVE_COLOR = '#388e3c'
    NEUTRAL_COLOR = '#424242'

    @classmethod
    def get_color_style(cls, val: float) -> str:
        if pd.isna(val):
            return ''
        weight = 'bold' if abs(val) > cls.BOLD_THRESHOLD_HIGH or abs(val) > cls.BOLD_THRESHOLD_LOW else 'normal'
        if val < 0:
            color = cls.NEGATIVE_COLOR
        elif val > 0:
            color = cls.POSITIVE_COLOR
        else:
            color = cls.NEUTRAL_COLOR
        return f'color: {color}; font-weight: {weight}'

    @classmethod
    def get_table_styles(cls) -> List[Dict]:
        return [
            {'selector': 'caption',
             'props': [('color', '#1976d2'), ('font-size', '18px'), ('font-weight', 'bold'),
                       ('text-align', 'center'), ('margin-bottom', '15px'), ('padding', '10px')]},
            {'selector': 'th',
             'props': [('background-color', '#1976d2'), ('color', 'white'), ('font-size', '14px'),
                       ('font-weight', 'bold'), ('text-align', 'center'), ('padding', '12px'),
                       ('border', '1px solid #ddd')]},
            {'selector': 'td',
             'props': [('text-align', 'center'), ('padding', '10px'), ('border', '1px solid #ddd'),
                       ('font-size', '13px')]},
            {'selector': 'th.row_heading',
             'props': [('background-color', '#f5f5f5'), ('color', '#333'), ('font-weight', 'bold'),
                       ('text-align', 'left'), ('padding', '10px'), ('border', '1px solid #ddd')]},
            {'selector': 'table',
             'props': [('border-collapse', 'collapse'), ('margin', '20px auto'),
                       ('box-shadow', '0 2px 8px rgba(0,0,0,0.1)'), ('border-radius', '8px'),
                       ('overflow', 'hidden')]}
        ]


class TableFormatter:
    def __init__(self, rowsConfig):
        self.row_manager = RowManager(rowsConfig)

    def _get_formatter_for_row(self, row_name: str) -> BaseFormatter:
        config = self.row_manager.get_row_config(row_name)
        if config:
            return FormatterFactory.get_formatter(config.format_type, config.custom_formatter)
        return FormatterFactory.get_formatter(FormatType.NUMBER)

    def _format_value(self, value: float, row_name: str, column_type: str = "data") -> str:
        config = self.row_manager.get_row_config(row_name)
        formatter = self._get_formatter_for_row(row_name)
        if column_type == "difference" and config and config.format_type == FormatType.PERCENTAGE:
            return formatter.format(value, with_sign=True)
        elif column_type == "difference_percent":
            percentage_formatter = FormatterFactory.get_formatter(FormatType.PERCENTAGE_WITH_SIGN)
            return percentage_formatter.format(value, with_sign=True)
        else:
            params = config.format_params if config and config.format_params else {}
            return formatter.format(value, **params)

    def apply_formatting(self, styled_table, tableau_final: pd.DataFrame) -> pd.DataFrame:
        display_to_tech = {v: k for k, v in self.row_manager.get_display_names_mapping().items()}
        formatted_data = {}
        for col in tableau_final.columns:
            col_values = []
            for idx in tableau_final.index:
                value = tableau_final.loc[idx, col]
                tech_name = display_to_tech.get(idx, idx)
                if col in ["Restés (Non)", "Partis (Oui)"]:
                    formatted_value = self._format_value(value, tech_name, "data")
                elif col == "Différence (Oui-Non)":
                    formatted_value = self._format_value(value, tech_name, "difference")
                elif col == "Différence (%)":
                    formatted_value = self._format_value(value, tech_name, "difference_percent")
                else:
                    formatted_value = str(value)
                col_values.append(formatted_value)
            formatted_data[col] = col_values
        formatted_df = pd.DataFrame(formatted_data, index=tableau_final.index)
        return formatted_df


class EmployeeAnalyzer():
    def __init__(self, rowsConfig):
        self.data_processor = DataProcessor()
        self.row_manager = RowManager(rowsConfig)
        self.style_config = StyleConfig()
        self.table_formatter = TableFormatter(rowsConfig)

    def analyze_and_display(self, df: pd.DataFrame) -> None:
        df_processed = self._prepare_data(df.copy())
        tableau_final = self._calculate_statistics(df_processed)
        self._display_results(tableau_final)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.data_processor.normalize_overtime_hours(df)
        df = self.data_processor.normalize_salary_increase(df)
        return df

    def _calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        row_numeric = self.row_manager.get_row_names()
        display_names = self.row_manager.get_display_names_mapping()

        comparatif = df.groupby("a_quitte_l_entreprise")[row_numeric].mean()

        tableau_final = pd.DataFrame({
            'Restés (Non)': comparatif.loc['Non'],
            'Partis (Oui)': comparatif.loc['Oui']
        })

        tableau_final['Différence (Oui-Non)'] = (
            tableau_final['Partis (Oui)'] - tableau_final['Restés (Non)']
        )

        # Remplacer l’index technique par les labels d’affichage
        tableau_final.index = [display_names[col] for col in tableau_final.index]

        # Différence en % (si la ligne est déjà en %, on laisse la diff brute)
        percentage_rows = self.row_manager.get_rows_by_format_type(FormatType.PERCENTAGE)
        tableau_final['Différence (%)'] = tableau_final.apply(
            lambda row: (
                row['Différence (Oui-Non)']
                if row.name in percentage_rows
                else (row['Différence (Oui-Non)'] / row['Restés (Non)'] * 100)
                if row['Restés (Non)'] != 0 else np.nan
            ),
            axis=1
        )

        # Trier
        tableau_final = tableau_final.reindex(
            tableau_final['Différence (%)'].abs().sort_values(ascending=False).index
        )

        return tableau_final

    def _display_results(self, tableau_final: pd.DataFrame) -> None:
        formatted_df = self.table_formatter.apply_formatting(None, tableau_final)

        styled = (
            formatted_df.style
            .set_caption("Étude comparative des salariés restants et quittant l'entreprise")
            .set_table_styles(self.style_config.get_table_styles())
        )

        def color_by_numeric(col: pd.Series) -> List[str]:
            numeric_series = tableau_final[col.name]
            return [self.style_config.get_color_style(v) for v in numeric_series]

        styled = styled.apply(color_by_numeric, axis=0, subset=["Différence (Oui-Non)", "Différence (%)"])
        display(styled)


def display_study(df: pd.DataFrame, rowsConfig) -> None:
    analyzer = EmployeeAnalyzer(rowsConfig)
    analyzer.analyze_and_display(df)