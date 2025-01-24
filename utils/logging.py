from datetime import datetime
import logging
from typing import Iterable

from pytorch_lightning.utilities import rank_zero_only as rzo
from rich.console import Console, ConsoleRenderable, RenderableType
from rich.containers import Renderables
from rich.logging import LogRender, RichHandler, FormatTimeCallable
from rich.table import Table
from rich.text import Text, TextType

__all__ = ["setup_logger"]


class _CustomLogRender(LogRender):

    def __call__(
        self,
        console: "Console",
        renderables: Iterable["ConsoleRenderable"],
        log_time: datetime | None = None,
        time_format: str | FormatTimeCallable | None = None,
        level: TextType = "",
        path: str | None = None,
        line_no: str | None = None,
        link_path: str | None = None,
    ) -> "Table":

        output = Table.grid(padding=(0, 1))
        output.expand = True
        output.add_column(ratio=1, style="log.message", overflow="fold")
        if self.show_time:
            output.add_column(style="log.time")
        if self.show_level:
            output.add_column(style="log.level", width=self.level_width)
        if self.show_path and path:
            output.add_column(style="log.path")

        row: list["RenderableType"] = []
        row.append(Renderables(renderables))
        if self.show_time:
            log_time = log_time or console.get_datetime()
            time_format = time_format or self.time_format
            if callable(time_format):
                log_time_display = time_format(log_time)
            else:
                log_time_display = Text(log_time.strftime(time_format))
            if log_time_display == self._last_time and self.omit_repeated_times:
                row.append(Text(" " * len(log_time_display)))
            else:
                row.append(log_time_display)
                self._last_time = log_time_display
        if self.show_level:
            row.append(level)
        if self.show_path and path:
            path_text = Text()
            path_text.append(path, style=f"link file://{link_path}" if link_path else "")
            if line_no:
                path_text.append(":")
                path_text.append(
                    f"{line_no}",
                    style=f"link file://{link_path}#{line_no}" if link_path else "",
                )
            row.append(path_text)

        output.add_row(*row)
        return output


class _CustomRichHandler(RichHandler):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._log_render = _CustomLogRender(
            show_time=self._log_render.show_time,
            show_level=self._log_render.show_level,
            show_path=self._log_render.show_path,
            time_format=self._log_render.time_format,
            omit_repeated_times=self._log_render.omit_repeated_times,
            level_width=None,
        )


def _get_console_width() -> int:
    """No less than 160 columns."""
    console = Console()
    if console.width > 160:
        return console.width
    return 160


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[_CustomRichHandler(
        console=Console(width=_get_console_width()),
        rich_tracebacks=True,
        show_path=False,
    )]
)


def setup_logger(name: str, rank_zero_only: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)

    if rank_zero_only:
        logger.debug = rzo(logger.debug)
        logger.info = rzo(logger.info)
        logger.warning = rzo(logger.warning)
        logger.error = rzo(logger.error)
        logger.critical = rzo(logger.critical)
    return logger
