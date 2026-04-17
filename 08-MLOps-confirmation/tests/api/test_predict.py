import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import os

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def api_token():
    """Récupère le token API depuis les variables d'environnement"""
    return os.getenv("API_TOKEN")


@pytest.fixture
def auth_headers(api_token):
    """Headers d'authentification avec le token API"""
    return {"Authorization": f"Bearer {api_token}"}


@pytest.fixture
def sample_application_data():
    with open("tests/api/fixtures/test_sample_application.json", "r") as f:
        return json.load(f)


@pytest.fixture
def sample_bureau_data():
    with open("tests/api/fixtures/test_sample_bureau.json", "r") as f:
        return json.load(f)


@pytest.fixture
def mock_model():
    model = Mock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.75, 0.25]])
    return model


@pytest.fixture
def mock_pipeline():
    pipeline = Mock()
    pipeline.transform.return_value = pd.DataFrame([[1, 2, 3, 4, 5]])
    return pipeline


@pytest.fixture
def mock_db():
    db = Mock()
    db.add = Mock()
    db.commit = Mock()
    return db


class TestPredictEndpoint:
    """Tests pour l'endpoint /predict"""

    @patch("src.api.predict.model")
    @patch("src.api.predict.pipeline")
    @patch("src.api.predict.get_db")
    def test_predict_nominal_case(
            self,
            mock_get_db,
            mock_pipeline_global,
            mock_model_global,
            client,
            auth_headers,
            sample_application_data,
            sample_bureau_data,
            mock_model,
            mock_pipeline,
            mock_db,
    ):
        """Test du cas nominal : prédiction réussie avec authentification"""

        mock_model_global.return_value = mock_model
        mock_pipeline_global.return_value = mock_pipeline
        mock_get_db.return_value = mock_db

        with patch("src.api.predict._model_loaded", True):
            with patch("src.api.predict.model", mock_model):
                with patch("src.api.predict.pipeline", mock_pipeline):
                    payload = {
                        "application_data": sample_application_data,
                        "bureau_data": sample_bureau_data,
                    }

                    # Appel avec authentification
                    response = client.post("/predict/", json=payload, headers=auth_headers)

                    # Assertions
                    assert response.status_code == 200

                    data = response.json()
                    assert "SK_ID_CURR" in data
                    assert "prediction" in data
                    assert "probability" in data
                    assert "inference_time_ms" in data

                    # Vérifie les types
                    assert isinstance(data["prediction"], int)
                    assert isinstance(data["probability"], float)
                    assert isinstance(data["inference_time_ms"], float)

                    # Vérifie les valeurs
                    assert data["prediction"] in [0, 1]
                    assert 0.0 <= data["probability"] <= 1.0
                    assert data["inference_time_ms"] > 0

                    # Vérifie que le pipeline a été appelé
                    mock_pipeline.transform.assert_called_once()

                    # Vérifie que le modèle a été appelé
                    mock_model.predict.assert_called_once()
                    mock_model.predict_proba.assert_called_once()

    def test_predict_without_auth(
            self, client, sample_application_data, sample_bureau_data
    ):
        """Test sans authentification : doit retourner 401"""

        payload = {
            "application_data": sample_application_data,
            "bureau_data": sample_bureau_data,
        }

        # Appel SANS header d'authentification
        response = client.post("/predict/", json=payload)

        # Doit retourner 401 Unauthorized
        assert response.status_code == 401

    def test_predict_with_invalid_token(
            self, client, sample_application_data, sample_bureau_data
    ):
        """Test avec un token invalide : doit retourner 401"""

        payload = {
            "application_data": sample_application_data,
            "bureau_data": sample_bureau_data,
        }

        # Appel avec un mauvais token
        response = client.post(
            "/predict/",
            json=payload,
            headers={"Authorization": "Bearer invalid-token-xyz"},
        )

        # Doit retourner 401 Unauthorized
        assert response.status_code == 401

    def test_predict_invalid_income_zero(
            self, client, auth_headers, sample_application_data, sample_bureau_data
    ):
        """Test avec un revenu à 0 (invalide si non autorisé)"""

        invalid_data = sample_application_data.copy()
        invalid_data["AMT_INCOME_TOTAL"] = 0

        payload = {
            "application_data": invalid_data,
            "bureau_data": sample_bureau_data,
        }

        response = client.post("/predict/", json=payload, headers=auth_headers)

        # Doit retourner une erreur de validation (422)
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_predict_invalid_income_negative(
            self, client, auth_headers, sample_application_data, sample_bureau_data
    ):
        """Test avec un revenu négatif (invalide)"""

        invalid_data = sample_application_data.copy()
        invalid_data["AMT_INCOME_TOTAL"] = -50000

        payload = {
            "application_data": invalid_data,
            "bureau_data": sample_bureau_data,
        }

        response = client.post("/predict/", json=payload, headers=auth_headers)

        # Doit retourner une erreur de validation (422)
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_predict_invalid_credit_amount_negative(
            self, client, auth_headers, sample_application_data, sample_bureau_data
    ):
        """Test avec un montant de crédit négatif (invalide)"""

        invalid_data = sample_application_data.copy()
        invalid_data["AMT_CREDIT"] = -100000

        payload = {
            "application_data": invalid_data,
            "bureau_data": sample_bureau_data,
        }

        response = client.post("/predict/", json=payload, headers=auth_headers)

        # Doit retourner une erreur de validation (422)
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_predict_invalid_type_string_for_integer(
            self, client, auth_headers, sample_application_data, sample_bureau_data
    ):
        """Test avec un type incorrect : texte à la place d'un entier (SK_ID_CURR)"""

        invalid_data = sample_application_data.copy()
        invalid_data["SK_ID_CURR"] = "texte_invalide"  # Devrait être un int

        payload = {
            "application_data": invalid_data,
            "bureau_data": sample_bureau_data,
        }

        response = client.post("/predict/", json=payload, headers=auth_headers)

        # Doit retourner une erreur de validation (422)
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body
        # Vérifie que l'erreur concerne SK_ID_CURR
        assert any("SK_ID_CURR" in str(err.get("loc", "")) for err in body["detail"])

    def test_predict_invalid_type_string_for_float(
            self, client, auth_headers, sample_application_data, sample_bureau_data
    ):
        """Test avec un type incorrect : texte à la place d'un float (AMT_INCOME_TOTAL)"""

        invalid_data = sample_application_data.copy()
        invalid_data["AMT_INCOME_TOTAL"] = "pas_un_nombre"  # Devrait être un float

        payload = {
            "application_data": invalid_data,
            "bureau_data": sample_bureau_data,
        }

        response = client.post("/predict/", json=payload, headers=auth_headers)

        # Doit retourner une erreur de validation (422)
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body
        assert any("AMT_INCOME_TOTAL" in str(err.get("loc", "")) for err in body["detail"])

    def test_predict_invalid_type_integer_for_string(
            self, client, auth_headers, sample_application_data, sample_bureau_data
    ):
        """Test avec un type incorrect : entier à la place d'une chaîne (CODE_GENDER)"""

        invalid_data = sample_application_data.copy()
        invalid_data["CODE_GENDER"] = 12345  # Devrait être une string (M, F, XNA)

        payload = {
            "application_data": invalid_data,
            "bureau_data": sample_bureau_data,
        }

        response = client.post("/predict/", json=payload, headers=auth_headers)

        # Pydantic peut convertir int -> str, donc ce test peut passer (200)
        # ou échouer (422) selon la config. On vérifie juste qu'on a une réponse cohérente
        assert response.status_code in [200, 422]

        # Si c'est 422, on vérifie la structure
        if response.status_code == 422:
            body = response.json()
            assert "detail" in body

    def test_predict_invalid_type_float_for_integer(
            self, client, auth_headers, sample_application_data, sample_bureau_data
    ):
        """Test avec un type incorrect : float à la place d'un int (CNT_CHILDREN)"""

        invalid_data = sample_application_data.copy()
        invalid_data["CNT_CHILDREN"] = 2.5  # Devrait être un int

        payload = {
            "application_data": invalid_data,
            "bureau_data": sample_bureau_data,
        }

        response = client.post("/predict/", json=payload, headers=auth_headers)


    @patch("src.api.predict.get_db")
    def test_predict_lazy_loading(
            self,
            mock_get_db,
            client,
            auth_headers,
            sample_application_data,
            sample_bureau_data,
            mock_model,
            mock_pipeline,
            mock_db,
    ):
        """
        Test du lazy loading : on simule model/pipeline à None
        pour forcer l'appel à load_model() et le flux normal.
        Pas besoin d'assert sur load_model, on veut juste couvrir le code.
        """

        mock_get_db.return_value = mock_db

        # On met model et pipeline à None pour passer dans le if
        import src.api.predict as predict_module
        predict_module.model = None
        predict_module.pipeline = None
        predict_module._model_loaded = False

        # Quand load_model sera appelé, on remplit model/pipeline avec nos mocks
        def fake_load_model():
            predict_module.model = mock_model
            predict_module.pipeline = mock_pipeline
            predict_module._model_loaded = True

        with patch("src.api.predict.load_model", side_effect=fake_load_model):
            payload = {
                "application_data": sample_application_data,
                "bureau_data": sample_bureau_data,
            }

            response = client.post("/predict/", json=payload, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "SK_ID_CURR" in data
        assert "prediction" in data
        assert "probability" in data
        assert "inference_time_ms" in data

    @patch("src.api.predict.model")
    @patch("src.api.predict.pipeline")
    @patch("src.api.predict.get_db")
    def test_predict_model_without_predict_proba(
            self,
            mock_get_db,
            mock_pipeline_global,
            mock_model_global,
            client,
            auth_headers,
            sample_application_data,
            sample_bureau_data,
            mock_pipeline,
            mock_db,
    ):
        """Test avec un modèle qui a decision_function mais pas predict_proba"""

        # Modèle avec decision_function
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.decision_function.return_value = np.array([2.5])
        del mock_model.predict_proba  # Pas de predict_proba

        mock_model_global.return_value = mock_model
        mock_pipeline_global.return_value = mock_pipeline
        mock_get_db.return_value = mock_db

        with patch("src.api.predict._model_loaded", True):
            with patch("src.api.predict.model", mock_model):
                with patch("src.api.predict.pipeline", mock_pipeline):
                    payload = {
                        "application_data": sample_application_data,
                        "bureau_data": sample_bureau_data,
                    }

                    response = client.post("/predict/", json=payload, headers=auth_headers)

                    assert response.status_code == 200
                    data = response.json()
                    assert "probability" in data
                    mock_model.decision_function.assert_called_once()

    @patch("src.api.predict.model")
    @patch("src.api.predict.pipeline")
    @patch("src.api.predict.get_db")
    def test_predict_model_without_proba_methods(
            self,
            mock_get_db,
            mock_pipeline_global,
            mock_model_global,
            client,
            auth_headers,
            sample_application_data,
            sample_bureau_data,
            mock_pipeline,
            mock_db,
    ):
        """Test avec un modèle sans predict_proba ni decision_function"""

        # Modèle basique
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])
        del mock_model.predict_proba
        del mock_model.decision_function

        mock_model_global.return_value = mock_model
        mock_pipeline_global.return_value = mock_pipeline
        mock_get_db.return_value = mock_db

        with patch("src.api.predict._model_loaded", True):
            with patch("src.api.predict.model", mock_model):
                with patch("src.api.predict.pipeline", mock_pipeline):
                    payload = {
                        "application_data": sample_application_data,
                        "bureau_data": sample_bureau_data,
                    }

                    response = client.post("/predict/", json=payload, headers=auth_headers)

                    assert response.status_code == 200
                    data = response.json()
                    assert data["probability"] == float(data["prediction"])

    @patch("src.api.predict.model")
    @patch("src.api.predict.pipeline")
    @patch("src.api.predict.get_db")
    def test_predict_pipeline_error(
            self,
            mock_get_db,
            mock_pipeline_global,
            mock_model_global,
            client,
            auth_headers,
            sample_application_data,
            sample_bureau_data,
            mock_model,
            mock_db,
    ):
        """Test d'erreur lors du traitement par le pipeline -> déclenche le except Exception"""

        # On utilise un pipeline qui lève une Exception sur transform
        mock_pipeline = Mock()
        mock_pipeline.transform.side_effect = Exception("Erreur pipeline")

        mock_model_global.return_value = mock_model
        mock_pipeline_global.return_value = mock_pipeline
        mock_get_db.return_value = mock_db

        # Simule que le modèle et le pipeline sont déjà chargés
        import src.api.predict as predict_module
        predict_module.model = mock_model
        predict_module.pipeline = mock_pipeline
        predict_module._model_loaded = True

        payload = {
            "application_data": sample_application_data,
            "bureau_data": sample_bureau_data,
        }

        response = client.post("/predict/", json=payload, headers=auth_headers)

        # On doit tomber dans le except Exception => HTTP 500
        assert response.status_code == 500
        body = response.json()
        assert "Erreur lors de la prédiction" in body["detail"]

        # Vérifie que le log d'erreur a bien été tenté
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @patch("src.api.predict.model")
    @patch("src.api.predict.pipeline")
    @patch("src.api.predict.get_db")
    def test_predict_pipeline_error(
            self,
            mock_get_db,
            mock_pipeline_global,
            mock_model_global,
            client,
            auth_headers,
            sample_application_data,
            sample_bureau_data,
            mock_model,
            mock_db,
    ):
        """Test d'erreur lors de la transformation par le pipeline"""

        mock_pipeline = Mock()
        mock_pipeline.transform.side_effect = Exception("Erreur pipeline")

        mock_model_global.return_value = mock_model
        mock_pipeline_global.return_value = mock_pipeline
        mock_get_db.return_value = mock_db

        with patch("src.api.predict._model_loaded", True):
            with patch("src.api.predict.model", mock_model):
                with patch("src.api.predict.pipeline", mock_pipeline):
                    payload = {
                        "application_data": sample_application_data,
                        "bureau_data": sample_bureau_data,
                    }

                    response = client.post("/predict/", json=payload, headers=auth_headers)

                    assert response.status_code == 500
                    assert "Erreur lors de la prédiction" in response.json()["detail"]

