import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import ApiConfig from './apiConfig';
import { AgentResponse } from './agent.types';

@Injectable({
  providedIn: 'root',
})
export class AgentService {
  constructor(private http: HttpClient) {}

  getSuggestions(fen: string): Observable<AgentResponse> {
    return this.http.get<AgentResponse>(
      `${ApiConfig.URL}agent/${encodeURIComponent(fen)}`
    );
  }
}
