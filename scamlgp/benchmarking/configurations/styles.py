# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import matplotlib

from scamlgp.optimizer import ScaMLGPBO

CMAP = matplotlib.cm.get_cmap("tab10")

OPTIMIZER_STYLES = {ScaMLGPBO: {"color": CMAP(0)[:3], "label": "ScaML-GP", "line": "-"}}
