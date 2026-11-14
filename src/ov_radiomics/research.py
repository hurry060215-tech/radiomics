"""Strictly keyed research tables, using only the common native algorithms."""
from dataclasses import dataclass
from ._native_common import request
from ._types import FeatureTable


@dataclass
class ResearchResult:
    table: FeatureTable
    provenance: list

    @property
    def features(self):
        return self.table.features

    def save(self,path,*,overwrite=False):
        return self.table.save(path,overwrite=overwrite)


def concat(sources,*,key_fields,varying_fields=(),already_registered):
    """Exact key populations; source/feature order is explicit insertion order.

    Registration is a caller declaration. This function performs no registration.
    Only explicitly permitted modality/region metadata may differ across sources.
    """
    values=[dict(prefix=name,table=table.to_dict() if isinstance(table,FeatureTable) else table)
        for name,table in sources.items()]
    output=request('research_concat',sources=values,config=dict(key_fields=list(key_fields),
        varying_fields=list(varying_fields),already_registered=already_registered))
    return ResearchResult(FeatureTable._from_native(output['table']),output['rows'])


def delta(table,*,baseline,followup,zero_denominator='error'):
    """Pair patient/lesion/modality/region; differences, ratios, relative fractions."""
    output=request('research_delta',table=table.to_dict() if isinstance(table,FeatureTable) else table,
        config=dict(baseline=baseline,followup=followup,zero_denominator=zero_denominator))
    return ResearchResult(FeatureTable._from_native(output['table']),output['pairs'])
