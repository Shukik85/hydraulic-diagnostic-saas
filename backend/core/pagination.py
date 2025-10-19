# core/pagination.py
# ОПТИМИЗИРОВАННАЯ ПАГИНАЦИЯ

from collections import OrderedDict

from django.core.paginator import InvalidPage

from rest_framework.exceptions import NotFound
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from rest_framework.utils.urls import remove_query_param, replace_query_param


class StandardResultsSetPagination(PageNumberPagination):
    """
    Стандартная пагинация для большинства API endpoints
    Оптимизирована для производительности
    """

    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100
    page_query_param = "page"

    def get_paginated_response(self, data):
        """
        Возвращает оптимизированный ответ с метаданными пагинации
        """
        return Response(
            OrderedDict(
                [
                    (
                        "links",
                        OrderedDict(
                            [
                                ("next", self.get_next_link()),
                                ("previous", self.get_previous_link()),
                            ]
                        ),
                    ),
                    ("count", self.page.paginator.count),
                    ("total_pages", self.page.paginator.num_pages),
                    ("current_page", self.page.number),
                    ("page_size", self.get_page_size(self.request)),
                    ("results", data),
                ]
            )
        )

    def paginate_queryset(self, queryset, request, view=None):
        """
        Оптимизированная пагинация с проверкой ошибок
        """
        page_size = self.get_page_size(request)
        if not page_size:
            return None

        paginator = self.django_paginator_class(queryset, page_size)
        page_number = self.get_page_number(request, paginator)

        try:
            self.page = paginator.page(page_number)
        except InvalidPage as exc:
            msg = self.invalid_page_message.format(
                page_number=page_number, message=str(exc)
            )
            raise NotFound(msg)

        if paginator.num_pages > 1 and self.template is not None:
            self.display_page_controls = True

        self.request = request
        return list(self.page)


class LargeResultsSetPagination(PageNumberPagination):
    """
    Пагинация для больших наборов данных (логи, отчеты)
    """

    page_size = 50
    page_size_query_param = "page_size"
    max_page_size = 200

    def get_paginated_response(self, data):
        return Response(
            OrderedDict(
                [
                    (
                        "links",
                        OrderedDict(
                            [
                                ("next", self.get_next_link()),
                                ("previous", self.get_previous_link()),
                                ("first", self.get_first_link()),
                                ("last", self.get_last_link()),
                            ]
                        ),
                    ),
                    (
                        "pagination",
                        OrderedDict(
                            [
                                ("count", self.page.paginator.count),
                                ("total_pages", self.page.paginator.num_pages),
                                ("current_page", self.page.number),
                                ("page_size", self.get_page_size(self.request)),
                                ("has_next", self.page.has_next()),
                                ("has_previous", self.page.has_previous()),
                            ]
                        ),
                    ),
                    ("results", data),
                ]
            )
        )

    def get_first_link(self):
        if not self.page.has_previous():
            return None
        url = self.request.build_absolute_uri()
        return remove_query_param(url, self.page_query_param)

    def get_last_link(self):
        if not self.page.has_next():
            return None
        url = self.request.build_absolute_uri()
        return replace_query_param(
            url, self.page_query_param, self.page.paginator.num_pages
        )


class SmallResultsSetPagination(PageNumberPagination):
    """
    Маленькая пагинация для малых наборов данных
    """

    page_size = 10
    page_size_query_param = "page_size"
    max_page_size = 50

    def get_paginated_response(self, data):
        return Response(
            OrderedDict(
                [
                    ("count", self.page.paginator.count),
                    ("next", self.get_next_link()),
                    ("previous", self.get_previous_link()),
                    ("results", data),
                ]
            )
        )


class NoPagination(PageNumberPagination):
    """
    Отключение пагинации для малых списков
    ОСТОРОЖНО: используйте только для гарантированно малых результатов!
    """

    page_size = None


class CursorPaginationOptimized(PageNumberPagination):
    """
    Оптимизированная cursor-based пагинация для больших данных
    Лучшая производительность для очень больших таблиц
    """

    page_size = 25
    ordering = "-created_at"  # Обязательная сортировка
    cursor_query_param = "cursor"
    cursor_query_description = "The pagination cursor value."
    page_size_query_param = "page_size"
    invalid_cursor_message = "Invalid cursor"
    max_page_size = 100

    def get_paginated_response(self, data):
        return Response(
            OrderedDict(
                [
                    (
                        "links",
                        OrderedDict(
                            [
                                ("next", self.get_next_link()),
                                ("previous", self.get_previous_link()),
                            ]
                        ),
                    ),
                    (
                        "meta",
                        OrderedDict(
                            [
                                ("page_size", self.get_page_size(self.request)),
                                ("ordering", self.ordering),
                            ]
                        ),
                    ),
                    ("results", data),
                ]
            )
        )
