import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from scipy import stats

N_AMOSTRA = 1042

dados_brutos = {}

try:
    dados_brutos = pd.read_csv('listings.csv', encoding="utf-8")
except UnicodeDecodeError:
    dados_brutos= pd.read_csv('listings.csv', encoding="latin-1")


def normalizar_texto(valor):
    """Remove acentos, espaços e altera para minúsculas."""
    valor = valor.astype("string").str.normalize("NFKD")
    valor = valor.str.encode("ascii", "ignore").str.decode("ascii")
    return valor


def limpar_dados(df):
    df = df.copy()

    for coluna in ["neighbourhood"]:
        df[coluna] = normalizar_texto(df[coluna])

    df = df.drop(columns=["id", "host_profile_id", "license"])
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    frequencias = df["neighbourhood"].value_counts()
    bairros_raros = frequencias[frequencias < 5].index

    df["neighbourhood"] = df["neighbourhood"].replace(
        {bairro: "Outros" for bairro in bairros_raros}
    )

    df = df.drop(df['price'].nlargest(2).index)

    return df


def exibir_exploracao_dataset(df):
    """Mostra medidas de qualidade, estatísticas e distribuições do dataset."""
    st.header("Visão geral do conjunto de dados processado")

    total_nulos = int(df.isna().sum().sum())
    identificadores = {"id", "host_id", "host_profile_id"}
    colunas_numericas = [
        c for c in df.select_dtypes(include="number").columns if c not in identificadores
    ]
    total_zeros = int(df[colunas_numericas].eq(0).sum().sum()) if colunas_numericas else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anúncios", f"{len(df):,}")
    c2.metric("Variáveis", f"{df.shape[1]:,}")
    c3.metric("Valores nulos", f"{total_nulos:,}")
    c4.metric("Valores zero (numéricas)", f"{total_zeros:,}")

    qualidade = pd.DataFrame({
        "Tipo": df.dtypes.astype(str),
        "Nulos": df.isna().sum(),
        "Nulos (%)": (df.isna().mean() * 100).round(2),
    })
    if colunas_numericas:
        qualidade["Zeros"] = df[colunas_numericas].eq(0).sum().reindex(df.columns).fillna(0).astype(int)
    else:
        qualidade["Zeros"] = 0
    st.subheader("Completude e tipo das variáveis")
    st.dataframe(qualidade, **LARGURA)

    with st.expander("Indicadores descritivos"):
        st.markdown("Medidas quantitativas: contagem, média, mediana, desvio-padrão, variância, quartis e amplitude.")
        st.dataframe(df.drop(columns=['host_id', 'latitude', 'longitude']).describe(exclude=['O']).round(2), hide_index=False, **LARGURA)

        st.markdown("Frequências e proporções das variáveis qualitativas:")
        categoricas = [
            c for c in df.select_dtypes(include=["object", "string", "category", "bool"]).columns
            if c != "last_review"
        ]
        if categoricas:
            variavel_qualitativa = st.selectbox("Variável qualitativa", categoricas, key=len(df))
            limite = st.slider("Categorias exibidas", min_value=5, max_value=50, value=20, key=f"limite_{len(df)}")
            valores = df[variavel_qualitativa].astype("string").fillna("(nulo)")
            distribuicao = valores.value_counts(dropna=False).rename("Frequencia").to_frame()
            distribuicao["Proporcao (%)"] = (distribuicao["Frequencia"] / len(df) * 100).round(2)
            distribuicao.index.name = "Categoria"
            st.dataframe(distribuicao.head(limite), **LARGURA)
            if len(distribuicao) > limite:
                st.caption(f"Exibindo as {limite} categorias mais frequentes de {len(distribuicao)}; as proporções usam todos os registros.")

    st.subheader("Distribuição de uma variável")
    numericas_grafico = colunas_numericas
    categoricas_grafico = [
        c for c in df.select_dtypes(include=["object", "string", "category", "bool"]).columns
        if 2 <= df[c].nunique(dropna=True) <= 200
    ]
    datas_grafico = [c for c in df.columns if c == "last_review"]
    opcoes = numericas_grafico + categoricas_grafico + datas_grafico

    if not opcoes:
        st.info("Não há variáveis adequadas para exibir distribuições.")
    else:
        variavel = st.selectbox("Selecione a variável", opcoes, key=f"variavel_{len(df)}")
        serie = df[variavel]
        if variavel in numericas_grafico:
            bins = st.slider("Número de intervalos", min_value=5, max_value=60, value=30, key=f"bins_{len(df)}")
            col_a, col_b = st.columns(2)
            col_a.plotly_chart(px.histogram(df, x=variavel, nbins=bins, title=f"Distribuição de {variavel}"), **LARGURA)
            col_b.plotly_chart(px.box(df, x=variavel, title=f"Boxplot de {variavel}"), **LARGURA)
        elif variavel in datas_grafico:
            datas = pd.to_datetime(serie, errors="coerce").dropna()
            if datas.empty:
                st.info("A variável não possui datas válidas para plotar.")
            else:
                por_ano = datas.dt.year.value_counts().sort_index().rename_axis("Ano").reset_index(name="Anúncios")
                st.plotly_chart(px.bar(por_ano, x="Ano", y="Anúncios", title="Anúncios por ano da última avaliação"), **LARGURA)
        else:
            frequencias = serie.astype("string").fillna("(nulo)").value_counts().head(20).rename_axis(variavel).reset_index(name="Anúncios")
            st.plotly_chart(px.bar(frequencias, x="Anúncios", y=variavel, orientation="h", title=f"Categorias mais frequentes: {variavel}"), **LARGURA)

    if len(numericas_grafico) >= 2:
        st.subheader("Relação entre duas variáveis numéricas")
        x_default = "price" if "price" in numericas_grafico else numericas_grafico[0]
        y_default = "number_of_reviews" if "number_of_reviews" in numericas_grafico else numericas_grafico[1]
        c_x, c_y = st.columns(2)
        eixo_x = c_x.selectbox("Eixo X", numericas_grafico, index=numericas_grafico.index(x_default), key=f"eixo_x_{len(df)}")
        opcoes_y = [c for c in numericas_grafico if c != eixo_x]
        eixo_y = c_y.selectbox("Eixo Y", opcoes_y, index=opcoes_y.index(y_default) if y_default in opcoes_y else 0, key=f"eixo_y_{len(df)}")
        st.plotly_chart(px.scatter(df, x=eixo_x, y=eixo_y, opacity=0.45, title=f"{eixo_y} por {eixo_x}"), **LARGURA)


# Streamlit 1.50+ aceita width="stretch"; versões anteriores usam use_container_width.
versao_streamlit = tuple(int(parte) for parte in st.__version__.split(".")[:2])
LARGURA = {"width": "stretch"} if versao_streamlit >= (1, 50) else {"use_container_width": True}

st.set_page_config(
    page_title="IA003 — Anúncios de estadia em Airbnb na cidade de São Paulo",
    page_icon="🏠",
    layout="wide",
)

st.title("IA003 — Anúncios de estadia em Airbnb na cidade de São Paulo")
st.caption("Uma interface de exploração dos anúncios de Airbnb em São Paulo, com Streamlit. Por Gabriel Leal")


tab1, tab2, tab3 = st.tabs(["Análise descritiva da população", "Análise descritiva da amostra", "Inferência estatística e comparação dos resultados"])
df = limpar_dados(dados_brutos)

with tab1:

    st.header("Definição do conjunto de dados")
    st.markdown(
        "Para este trabalho foi selecionado o conjunto de dados dos anúncios de estadia em Airbnb na cidade de São Paulo, disponibilizado pela própria plataforma (Inside Airbnb, https:\\/\\/insideairbnb.com\\/get-the-data, licença CC BY 4.0). " \
        "O conjunto de dados é atualizado constantemente e contém informações sobre localização, preço, tipo de quarto e outras características dos anúncios. A versão utilizada" \
        " é de 14 de Setembro de 2026, e possui " + str(len(df)) + " registros e 19 variáveis. Abaixo é possível visualizar o conjunto completo de dados:" \
    )
    
    with st.expander("Visualizar o conjunto completo de dados"):
            st.dataframe(dados_brutos, hide_index=True, **LARGURA)

    st.markdown(
        "**Variáveis e tipos:** \n\nIdentificadores: `id`, `host_id`, `host_profile_id`; \n\n" 
        "Categoricas/textuais: `name`, `host_name`, `neighbourhood_group`, `neighbourhood`, `room_type`, `license`; \n\n"
        "Geográficas continuas: `latitude`, `longitude`; \n\nQuantitativas continuas: `price`, `reviews_per_month`; \n\n"
        "Quantitativas discretas: `minimum_nights`, `number_of_reviews`, `calculated_host_listings_count`, "
        "`availability_365`, `number_of_reviews_ltm`; \n\n"
        "Temporal: `last_review`. \n\n"
        f"**Qualidade dos dados ({len(dados_brutos):,} registros):** \n\n`license` tem 44.080 nulos; `price`, 1.539; "
        "`minimum_nights`, 113; `last_review` e `reviews_per_month`, 5.810 cada. \n\nHá zeros em "
        "`number_of_reviews` (5.810), `availability_365` (908) e `number_of_reviews_ltm` (8.715); "
        "`price` nao tem zeros."
    )

    st.header("Tratamento dos dados")
    st.markdown(
        "O tratamento dos dados incluiu a remoção de colunas não utilizadas, o preenchimento de valores nulos e a normalização de variáveis. "
        "A coluna `license` foi removida por conter somente valores nulos. A coluna `reviews_per_month` foi preenchida com zeros para os registros que tinham valores nulos."
        " Já as colunas `id` e `host_profile_id` foram removidas por não serem relevantes para a análise. "
        " Duas linhas com valores de `price` muito altos foram removidas, pois distorciam a análise. Uma delas tinha um preço no valor de R\\$500.210, e a outra de R\\$200.399, o que as fazia destoar do restante dos anúncios. A coluna `neighbourhood` foi normalizada para remover acentos e espaços, e os bairros com menos de 5 anúncios foram agrupados na categoria `Outros`."
    )

    with st.expander("Visualizar o conjunto de dados processado"):
                st.dataframe(df, hide_index=True, **LARGURA)

    st.markdown(
        "Abaixo, é possível visualizar as estatísticas descritivas do conjunto de dados processado, com a exclusão das colunas `host_id`, `latitude` e `longitude`, que não são relevantes para a análise."
        "Para cada variável, são apresentadas as medidas de tendência central, dispersão e distribuição, como média, mediana, desvio padrão, mínimo, máximo e quartis."
    )

    exibir_exploracao_dataset(df)

with tab2:
    st.header("Análise descritiva da amostra")
    st.markdown(
        "Nesta seção, são apresentados os resultados da análise descritiva realizada sobre a amostra selecionada do conjunto de dados processado. "
        "\n\nPara a seleção da amostra, foi utilizado um tamanho de amostra de 1.042 registros, utilizando a função `sample` do pandas, com uma semente aleatória de 42 para garantir a reprodutibilidade dos resultados. "
        "A amostra foi selecionada de forma aleatória, garantindo que todos os registros do conjunto de dados processado tivessem a mesma probabilidade de serem incluídos na amostra. "
        "O tamanho selecionado para a amostra foi calculado utilizando a ferramenta https:\\/\\/pt.surveymonkey.com\\/learn\\/research-and-analysis\\/sample-size-calculator\\/, nela"  
        " foram usados os seguintes parâmetros: tamanho da população = 44.078, margem de erro = 3%, nível de confiança = 95%. "
    )

    df_amostra = df.sample(n=N_AMOSTRA, random_state=42)
    # 2. Definindo parâmetros do teste
    sucessos = (df_amostra["neighbourhood_group"] == "Se").sum()  # Anúncios no Centro
    n = len(df_amostra)  # Tamanho total da amostra
    p_hat = sucessos / n  # Proporção observada na amostra
    p_nulo = 0.25  # Valor sob a Hipótese Nula (H0)
    alpha = 0.05  # Nível de significância (5%)

    exibir_exploracao_dataset(df_amostra)


with tab3: 
    
    st.header("Inferência para parâmetro de proporção")
    st.subheader(f"Explorando a proporção de anúncios localizados no Centro (Hipótese nula: p = 25%)")
    st.markdown(
        "Nesta seção, são apresentados os resultados da análise inferencial realizada sobre o conjunto de dados processado. "
        "Foram aplicados testes estatísticos para verificar a significância das diferenças observadas entre as variáveis, bem como para estimar intervalos de confiança para as proporções e médias das variáveis de interesse."
        "\n\nO teste de proporção foi aplicado para verificar se a proporção de anúncios localizados no Centro é significativamente diferente de 25%, com um nível de significância de 5%. "
        "Para esse teste foi considerada a variável `neighbourhood_group`, com o valor 'Se' indicando a localização do anúncio nos subdistritos da cidade que representa a região central de São Paulo. Então, foi calculada a proporção de anúncios localizados no Centro em relação ao total de anúncios na amostra. "
        "\n\nO teste de proporção foi realizado utilizando a função `proportions_ztest` da biblioteca `statsmodels`, que calcula a estatística de teste Z e o valor-p associado. "
        "A ideia foi explorar e comparar a proporção de anúncios localizados no Centro com a proporção hipotética de 25%, verificando se há evidências estatísticas suficientes para rejeitar a hipótese nula. Por poder indicar uma concentração de anúncios próximos aos principais pontos de acesso central da cidade, a proporção de anúncios no Centro é uma métrica relevante para avaliar a distribuição geográfica dos anúncios de estadia em Airbnb na cidade de São Paulo."
    )
    st.markdown(
        "#### Passo a passo da inferência de proporção realizada \n"
        "1. **Variável e sucesso:** `neighbourhood_group`; consideramos sucesso cada anúncio com valor `Se`, categoria definida como Centro.\n"
        "2. **Hipóteses:** H₀: p = 0,25; H₁: p ≠ 0,25. O teste é bilateral e usa α = 0,05.\n"
        "3. **Amostra:** conta-se `x` anúncios do grupo Se entre `1042` registros e calculamos p̂ = x/n.\n"
        "4. **Estatística de teste:** O teste Z de uma proporção fornece Z e o valor-p bilateral.\n"
        "5. **Intervalo de confiança:** é estimada a proporção populacional, com nível de confiança de 95%.\n"
        "6. **Decisão:** se valor-p < 0,05, rejeitamos H₀; caso contrário, não rejeitamos H₀. O intervalo indica valores plausíveis para a proporção populacional."
    )    


    # 3. Executando o Teste Z
    stat, p_value = proportions_ztest(
        count=sucessos, nobs=n, value=p_nulo, alternative="two-sided"
    )

    # 4. Calculando o Intervalo de Confiança (95%)
    ci_inf, ci_sup = proportion_confint(
        count=sucessos, nobs=n, alpha=alpha, method="normal"
    )

    # 5. Exibindo o Relatório Formatado
    print("=== RELATÓRIO ESTATÍSTICO: PROPORÇÃO DE ANÚNCIOS NO CENTRO ===")
    print(f"Tamanho da Amostra (n): {n}")
    print(f"Anúncios no Centro (Sucessos): {sucessos}")
    print(f"Proporção Amostral Observada (p^): {p_hat:.4f} ({p_hat*100:.2f}%)")
    print(f"Proporção Hipotética (H0): {p_nulo:.2f} ({p_nulo*100:.2f}%)")
    print("-" * 50)
    print(f"Nível de Significância (alfa): {alpha}")
    print(f"Estatística de Teste (Z): {stat:.4f}")
    print(f"Valor-p (p-value): {p_value:.6f}")
    print(
        f"Intervalo de Confiança ({int((1-alpha)*100)}%): [{ci_inf:.4f}, {ci_sup:.4f}] -> [{ci_inf*100:.2f}%, {ci_sup*100:.2f}%]"
    )
    print("-" * 50)

    # 6. Conclusão automática
    if p_value < alpha:
        print(
            "Conclusão: Rejeitamos H_0. Há evidências estatísticas significativas de que"
        )
        print("a proporção real de anúncios no Centro é diferente de 25%.")
    else:
        print(
            "Conclusão: Não rejeitamos H_0. Não há evidências suficientes para afirmar"
        )
        print("que a proporção real difere de 25%.")

        # 4. Layout Principal do Dashboard: Exibindo Métricas
    

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Total de Imóveis (n)", value=f"{n:,}")
    col2.metric(label="Proporção Amostral (p̂)", value=f"{p_hat:.2%}")
    col3.metric(label="Estatística Z", value=f"{stat:.2f}")
    col4.metric(
        label="Valor-p",
        value=f"{p_value:.4f}",
        delta="Significativo" if p_value < alpha else "Não Significativo",
        delta_color="inverse" if p_value < alpha else "off",
    )

    st.markdown("---")

    # 5. Exibição Detalhada: Intervalo de Confiança e Conclusão Interativa
    col_ic, col_conc = st.columns(2)

    with col_ic:
        st.markdown(f"### 📉 Intervalo de Confiança ({int((1-alpha)*100)}%)")
        st.info(
            f"Com {int((1-alpha)*100)}% de confiança, a verdadeira proporção da"
            f" população para Centro está entre **{ci_inf*100:.2f}%** e"
            f" **{ci_sup*100:.2f}%**."
        )

    with col_conc:
        st.markdown("### 🎯 Conclusão Estatística")
        if p_value < alpha:
            st.success(
                f"**Decisão: Rejeitamos a Hipótese Nula (H₀)**\n\nComo o valor-p"
                f" ({p_value:.4f}) é menor que α ({alpha}), há evidências estatísticas"
                f" suficientes para afirmar que a proporção real de anúncios no"
                f" Centro difere de **{p_nulo*100:.1f}%**."
            )
        else:
            st.warning(
                f"**Decisão: Não rejeitamos a Hipótese Nula (H₀)**\n\nComo o valor-p"
                f" ({p_value:.4f}) é maior ou igual a α ({alpha}), não há evidências"
                f" suficientes para refutar que a proporção é de **{p_nulo*100:.1f}%**."
            )    

    st.markdown("### Comparação da população e da amostra")
    st.markdown(
        "A tabela compara a contagem e a proporção de anúncios em `neighbourhood_group = Se` "
        "(categoria usada como Centro nesta análise) no conjunto processado completo (`df`) e na amostra aleatória (`df_amostra`). "
        "A proporção amostral é a estimativa submetida ao teste Z e ao intervalo de confiança exibidos acima."

    )
    
    sucessos_pop = int((df["neighbourhood_group"] == "Se").sum())
    total_pop = len(df)
    proporcao_pop = sucessos_pop / total_pop if total_pop else 0
    tabela_proporcao = pd.DataFrame([
        {
            "Conjunto": "População observada (df)",
            "Total de anúncios": total_pop,
            "Anúncios no grupo Se": sucessos_pop,
            "Proporção no grupo Se": proporcao_pop,
        },
        {
            "Conjunto": "Amostra (df_amostra)",
            "Total de anúncios": n,
            "Anúncios no grupo Se": sucessos,
            "Proporção no grupo Se": p_hat,
        },
    ])
    tabela_proporcao["Proporção no grupo Se"] = tabela_proporcao["Proporção no grupo Se"].map(lambda valor: f"{valor:.2%}")
    st.dataframe(tabela_proporcao, hide_index=True, **LARGURA)
    st.markdown("---")
    st.header("Inferência para parâmetro de média")
    st.subheader("Comparação do preço médio: Itaim Bibi × Republica")
    st.markdown(
        "**Pergunta:** os anúncios de Itaim Bibi e Republica têm preços médios por noite diferentes? "
        "Considerando que ao analisar a população, os dois bairros foram os com mais anúncios, e que a amostra foi sorteada aleatoriamente, é possível estimar a diferença de preços médios entre os dois bairros e verificar se essa diferença é estatisticamente significativa. "
        "A ideia foi explorar o impacto da região/localização no preço dos anúncios, considerando que bairros diferentes podem ter características distintas que influenciam o valor cobrado pelos anfitriões e pela oferta. "
        "A variável quantitativa é `price` (R$), e os grupos são definidos por `neighbourhood`. "
        "Foi usada a amostra já sorteada (`N_AMOSTRA`, `random_state=42`) para estimar e testar a diferença; "
        "o conjunto completo fornece as médias observadas para comparação."
    )
    st.markdown(
        "### Passo a passo da hipótese\n"
        "1. **Parâmetro:** Δ = μ_Itaim − μ_Republica, diferença entre os preços médios.\n"
        "2. **Hipóteses:** H₀: Δ = 0 (médias iguais); H₁: Δ ≠ 0 (médias diferentes).\n"
        "3. **Amostra:** calcular média, desvio-padrão e tamanho de cada bairro em `df_amostra`.\n"
        "4. **Incerteza:** calcular o erro-padrão e o IC de 95% pelo teste t de Welch, que permite variâncias diferentes. "
        "Aplicamos correção para população finita porque o sorteio foi sem reposição.\n"
        "5. **Decisão:** comparar o valor-p com α = 0,05. Se valor-p < α, rejeitamos H₀."
    )

    bairro_a, bairro_b = "Itaim Bibi", "Republica"
    alpha_comparacao = 0.05

    def estatisticas_bairro(base, bairro):
        serie = pd.to_numeric(base.loc[base["neighbourhood"] == bairro, "price"], errors="coerce").dropna()
        return {
            "n": len(serie),
            "media": float(serie.mean()) if len(serie) else np.nan,
            "dp": float(serie.std(ddof=1)) if len(serie) > 1 else np.nan,
        }

    pop_a, pop_b = estatisticas_bairro(df, bairro_a), estatisticas_bairro(df, bairro_b)
    am_a, am_b = estatisticas_bairro(df_amostra, bairro_a), estatisticas_bairro(df_amostra, bairro_b)

    if min(pop_a["n"], pop_b["n"], am_a["n"], am_b["n"]) < 2:
        st.warning("A população ou a amostra não contém pelo menos dois preços válidos em cada bairro; não é possível calcular o teste de Welch.")
    else:
        # Correção para população finita por bairro, pois a amostra foi sorteada sem reposição de df.
        fpc_a = np.sqrt((pop_a["n"] - am_a["n"]) / (pop_a["n"] - 1)) if am_a["n"] < pop_a["n"] else 0.0
        fpc_b = np.sqrt((pop_b["n"] - am_b["n"]) / (pop_b["n"] - 1)) if am_b["n"] < pop_b["n"] else 0.0
        var_a = (am_a["dp"] ** 2 / am_a["n"]) * fpc_a ** 2
        var_b = (am_b["dp"] ** 2 / am_b["n"]) * fpc_b ** 2
        ep_delta = np.sqrt(var_a + var_b)
        gl_welch = (
            (var_a + var_b) ** 2 / (var_a ** 2 / (am_a["n"] - 1) + var_b ** 2 / (am_b["n"] - 1))
            if ep_delta > 0 else min(am_a["n"] - 1, am_b["n"] - 1)
        )
        delta_amostra = am_a["media"] - am_b["media"]
        delta_pop = pop_a["media"] - pop_b["media"]
        t_welch = delta_amostra / ep_delta if ep_delta > 0 else (0.0 if delta_amostra == 0 else np.inf)
        p_welch = 2 * stats.t.sf(abs(t_welch), gl_welch) if np.isfinite(t_welch) else 0.0
        tcrit = stats.t.ppf(1 - alpha_comparacao / 2, gl_welch)
        ic_delta = (delta_amostra - tcrit * ep_delta, delta_amostra + tcrit * ep_delta)

        st.markdown("### Resultados: amostra e população observada")
        tabela = pd.DataFrame([
            {"Conjunto": "Amostra", "Bairro": bairro_a, "n": am_a["n"], "Preço médio (R\\$)": am_a["media"]},
            {"Conjunto": "Amostra", "Bairro": bairro_b, "n": am_b["n"], "Preço médio (R\\$)": am_b["media"]},
            {"Conjunto": "População", "Bairro": bairro_a, "n": pop_a["n"], "Preço médio (R\\$)": pop_a["media"]},
            {"Conjunto": "População", "Bairro": bairro_b, "n": pop_b["n"], "Preço médio (R\\$)": pop_b["media"]},
        ])
        st.dataframe(tabela, hide_index=True, **LARGURA)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Diferença amostral (Itaim − Republica)", f"R\\$ {delta_amostra:,.2f}")
        m2.metric("Diferença na População", f"R\\$ {delta_pop:,.2f}")
        m3.metric("Estatística t de Welch", f"{t_welch:.3f}")
        m4.metric("Valor-p", f"{p_welch:.4g}")

        st.markdown(
            f"**IC de 95% para Δ na amostra:** [R\\$ {ic_delta[0]:,.2f}, R\\$ {ic_delta[1]:,.2f}]  \n"
            f"**Nível de significância:** α = {alpha_comparacao:.2f} · **Graus de liberdade aproximados (Welch):** {gl_welch:.1f}."
        )
        if p_welch < alpha_comparacao:
            st.success(f"Como valor-p ({p_welch:.4g}) < α ({alpha_comparacao:.2f}), rejeitamos H₀. A amostra fornece evidência de diferença entre os preços médios dos bairros.")
        else:
            st.info(f"Como valor-p ({p_welch:.4g}) ≥ α ({alpha_comparacao:.2f}), não rejeitamos H₀. A amostra não fornece evidência suficiente de diferença entre os preços médios.")

        direcao = "maior" if delta_amostra > 0 else "menor" if delta_amostra < 0 else "igual"
        st.markdown(
            f"### Exploração e interpretação\n"
            f"Na amostra, o preço médio de Itaim Bibi é **{direcao}** que o de Republica, com diferença de **R\\$ {delta_amostra:,.2f}**. "
            f"No conjunto completo da população, a diferença observada é **R\\$ {delta_pop:,.2f}**. "
            f"A diferença entre essas estimativas é **R\\$ {delta_amostra-delta_pop:,.2f}**, indicando o afastamento da amostra em relação à referência calculada com os registros disponíveis."
        )
        st.markdown("### Gráficos Q–Q dos preços na amostra")
        st.markdown(
            "Cada gráfico compara os quantis observados de `price` no bairro com os quantis teóricos de uma distribuição normal. "
            "Pontos próximos à linha indicam que a distribuição do grupo é aproximadamente normal; curvaturas e afastamentos nas extremidades "
            "podem indicar assimetria ou caudas pesadas. O teste de Welch não exige variâncias iguais, mas a normalidade (ou amostras suficientemente grandes) "
            "ajuda a justificar a inferência sobre as médias. Os gráficos usam os mesmos dados de `df_amostra` considerados no teste."
        )
        precos_a = pd.to_numeric(df_amostra.loc[df_amostra["neighbourhood"] == bairro_a, "price"], errors="coerce").dropna()
        precos_b = pd.to_numeric(df_amostra.loc[df_amostra["neighbourhood"] == bairro_b, "price"], errors="coerce").dropna()
        fig_qq, eixos_qq = plt.subplots(1, 2, figsize=(12, 5))
        stats.probplot(precos_a, dist="norm", plot=eixos_qq[0])
        eixos_qq[0].set_title(f"Q–Q plot — {bairro_a}")
        eixos_qq[0].set_xlabel("Quantis teóricos normais")
        eixos_qq[0].set_ylabel("Quantis observados de price (R$)")
        stats.probplot(precos_b, dist="norm", plot=eixos_qq[1])
        eixos_qq[1].set_title(f"Q–Q plot — {bairro_b}")
        eixos_qq[1].set_xlabel("Quantis teóricos normais")
        eixos_qq[1].set_ylabel("Quantis observados de price (R$)")
        fig_qq.tight_layout()
        st.pyplot(fig_qq, **LARGURA)
        plt.close(fig_qq)
